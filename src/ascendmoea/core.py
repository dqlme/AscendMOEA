"""Core population, problem, and algorithm abstractions."""

from __future__ import annotations

import copy
import math
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable

import torch

from .device import (
    configure_torch,
    real_dtype_for_device,
    safe_dtype,
    seed_everything,
    select_device,
)


def to_tensor(
    x, device: torch.device | None = None, dtype: torch.dtype | None = None
) -> torch.Tensor:
    """Convert data to a tensor on the requested device and safe dtype."""
    if isinstance(x, torch.Tensor):
        target_device = device if device is not None else x.device
        return x.to(device=target_device, dtype=safe_dtype(dtype, target_device))
    if device is None:
        device = torch.device("cpu")
    return torch.as_tensor(x, device=device, dtype=safe_dtype(dtype, device))


def to_numpy(x):
    """Convert a tensor to a detached CPU NumPy array and pass through other data."""
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return x


def round_half_away_from_zero(x: torch.Tensor) -> torch.Tensor:
    """Round values to the nearest integer with ties directed away from zero."""
    return torch.sign(x) * torch.floor(torch.abs(x) + 0.5)


@dataclass
class Population:
    """Hold batched decisions, objectives, and inequality constraints.

    Constraint values follow the convention ``cons <= 0`` for feasibility.
    Every tensor uses its first dimension as the population dimension.
    """

    decs: torch.Tensor
    objs: torch.Tensor
    cons: torch.Tensor

    def __post_init__(self):
        """Normalize fields and validate the population batch contract."""
        if not isinstance(self.decs, torch.Tensor):
            self.decs = to_tensor(self.decs)
        if not isinstance(self.objs, torch.Tensor):
            self.objs = to_tensor(self.objs, device=self.decs.device)
        elif self.objs.device != self.decs.device:
            self.objs = self.objs.to(self.decs.device)
        if not isinstance(self.cons, torch.Tensor):
            self.cons = to_tensor(self.cons, device=self.decs.device)
        elif self.cons.device != self.decs.device:
            self.cons = self.cons.to(self.decs.device)

        fields = {"decs": self.decs, "objs": self.objs, "cons": self.cons}
        for name, tensor in fields.items():
            if tensor.ndim != 2:
                raise ValueError(f"Population.{name} must be a two-dimensional tensor")
        rows = self.decs.shape[0]
        if self.objs.shape[0] != rows or self.cons.shape[0] != rows:
            raise ValueError("Population fields must have the same number of rows")
        if self.decs.shape[1] == 0 or self.objs.shape[1] == 0:
            raise ValueError("Population decisions and objectives must contain at least one column")

    @property
    def feasible(self) -> torch.Tensor:
        """Return a Boolean feasibility mask for all individuals."""
        return torch.all(self.cons <= 0, dim=1)

    def best(self) -> "Population":
        """Return the best feasible individual or nondominated subset."""
        feas = self.feasible
        if not torch.any(feas):
            return Population(self.decs[:0], self.objs[:0], self.cons[:0])
        sub = Population(self.decs[feas], self.objs[feas], self.cons[feas])
        if sub.objs.shape[1] == 1:
            i = torch.argmin(sub.objs[:, 0])
            return Population(sub.decs[[i]], sub.objs[[i]], sub.cons[[i]])
        nd = non_dominated_mask(sub.objs)
        return Population(sub.decs[nd], sub.objs[nd], sub.cons[nd])

    def __len__(self) -> int:
        """Return the number of individuals."""
        return int(self.decs.shape[0])

    def __getitem__(self, index) -> "Population":
        """Select individuals while preserving a two-dimensional batch."""
        decs = self.decs[index]
        objs = self.objs[index]
        cons = self.cons[index]
        if decs.ndim == 1:
            decs = decs.unsqueeze(0)
            objs = objs.unsqueeze(0)
            cons = cons.unsqueeze(0)
        return Population(decs, objs, cons)

    def clone(self) -> "Population":
        """Return a deep tensor clone of the population."""
        return Population(self.decs.clone(), self.objs.clone(), self.cons.clone())

    def detach(self) -> "Population":
        """Return a population detached from all autograd graphs."""
        return Population(self.decs.detach(), self.objs.detach(), self.cons.detach())

    def to(
        self,
        device: torch.device | str,
        dtype: torch.dtype | None = None,
    ) -> "Population":
        """Move all population tensors to a device and optional real dtype."""
        target = torch.device(device)
        real_dtype = safe_dtype(dtype or self.decs.dtype, target)
        return Population(
            self.decs.to(device=target, dtype=real_dtype),
            self.objs.to(device=target, dtype=real_dtype),
            self.cons.to(device=target, dtype=real_dtype),
        )


class Problem(ABC):
    """Base class for vectorized optimization problems.

    Subclasses normally implement :meth:`setting` and :meth:`cal_obj`; they
    may also implement :meth:`cal_con`, :meth:`get_optimum`, and :meth:`get_pf`.
    Objective and constraint functions must accept an ``(N, D)`` tensor and
    evaluate the complete batch without per-individual Python loops.
    """

    name = "Problem"

    def __init__(self, n: int = 100, m: int | None = 2, d: int | None = 10, max_fe: int = 10000):
        """Initialize dimensions, evaluation budget, and CPU-resident state.

        Args:
            n: Requested population size.
            m: Number of objectives. ``None`` lets :meth:`setting` choose it.
            d: Number of decision variables. ``None`` lets :meth:`setting` choose it.
            max_fe: Maximum number of objective-function evaluations.
        """
        if int(n) < 1:
            raise ValueError("n must be at least one")
        if m is not None and int(m) < 1:
            raise ValueError("m must be at least one")
        if d is not None and int(d) < 1:
            raise ValueError("d must be at least one")
        if int(max_fe) < 1:
            raise ValueError("max_fe must be at least one")
        self.N = int(n)
        self.M = None if m is None else int(m)
        self.D = None if d is None else int(d)
        self.max_fe = int(max_fe)
        self.FE = 0
        self._device = torch.device("cpu")
        self.dtype = real_dtype_for_device(self._device)
        self.lower = torch.empty(0, dtype=self.dtype, device=self.device)
        self.upper = torch.empty(0, dtype=self.dtype, device=self.device)
        self.encoding = torch.empty(0, dtype=torch.long, device=self.device)
        self._optimum = None
        self._pf = None
        self._feasible_obj = None
        self._con_dim = None
        self.setting()
        if self.M is None:
            self.M = 2
        if self.D is None:
            self.D = 10
        self.lower = self._bound_tensor(self.lower, 0.0)
        self.upper = self._bound_tensor(self.upper, 1.0)
        if self.encoding.numel() == 0:
            self.encoding = torch.ones(self.D, dtype=torch.long, device=self.device)
        else:
            self.encoding = torch.as_tensor(
                self.encoding, dtype=torch.long, device=self.device
            ).reshape(-1)
            if self.encoding.numel() == 1:
                self.encoding = self.encoding.expand(self.D).clone()
            elif self.encoding.numel() != self.D:
                raise ValueError(f"encoding must contain 1 or D={self.D} values")
        self._refresh_encoding_cache()

    @property
    def device(self) -> torch.device:
        """Return the device that owns problem data and evaluations."""
        return self._device

    @device.setter
    def device(self, value: torch.device | str) -> None:
        """Move problem state and cached tensors to a new device."""
        device = torch.device(value)
        self._device = device
        self.dtype = real_dtype_for_device(device)
        for name in ("lower", "upper", "encoding", "_optimum", "_pf", "_feasible_obj"):
            if not hasattr(self, name):
                continue
            tensor = getattr(self, name)
            if isinstance(tensor, torch.Tensor):
                dtype = self.dtype if tensor.is_floating_point() else tensor.dtype
                setattr(self, name, tensor.to(device=device, dtype=dtype))
        if hasattr(self, "encoding") and self.encoding.numel() > 0:
            self._refresh_encoding_cache()

    def _refresh_encoding_cache(self) -> None:
        """Cache masks and counts for the five supported encoding types."""
        if torch.any((self.encoding < 1) | (self.encoding > 5)):
            raise ValueError("encoding values must be integers from 1 through 5")
        codes = self.encoding.detach().cpu().tolist()
        self.encoding_masks = tuple(self.encoding == code for code in range(1, 6))
        self.encoding_counts = tuple(codes.count(code) for code in range(1, 6))
        self.real_or_integer_mask = self.encoding_masks[0] | self.encoding_masks[1]
        self.integer_or_label_mask = self.encoding_masks[1] | self.encoding_masks[2]
        self.integer_like_mask = self.integer_or_label_mask | self.encoding_masks[3]
        self.has_encoded_variables = any(self.encoding_counts[2:])

    @abstractmethod
    def setting(self) -> None:
        """Configure dimensions, bounds, encoding, and static problem data."""

    def initialization(self, n: int | None = None) -> Population:
        """Sample and evaluate an initial population.

        Args:
            n: Number of individuals. Defaults to the configured population size.

        Returns:
            The initialized and evaluated population.
        """
        n = self.N if n is None else int(n)
        if n < 1:
            raise ValueError("initial population size must be at least one")
        dec = torch.zeros((n, self.D), dtype=self.dtype, device=self.device)
        real = self.encoding_masks[0]
        integer_or_label = self.integer_or_label_mask
        binary = self.encoding_masks[3]
        permutation = self.encoding_masks[4]
        if self.encoding_counts[0]:
            dec[:, real] = (
                torch.rand((n, self.encoding_counts[0]), dtype=self.dtype, device=self.device)
                * (self.upper[real] - self.lower[real])
                + self.lower[real]
            )
        integer_or_label_count = self.encoding_counts[1] + self.encoding_counts[2]
        if integer_or_label_count:
            values = (
                torch.rand(
                    (n, integer_or_label_count),
                    dtype=self.dtype,
                    device=self.device,
                )
                * (self.upper[integer_or_label] - self.lower[integer_or_label])
                + self.lower[integer_or_label]
            )
            dec[:, integer_or_label] = round_half_away_from_zero(values)
        if self.encoding_counts[3]:
            count = self.encoding_counts[3]
            dec[:, binary] = torch.randint(0, 2, (n, count), device=self.device).to(self.dtype)
            self.lower[binary] = 0
            self.upper[binary] = 1
        if self.encoding_counts[4]:
            count = self.encoding_counts[4]
            dec[:, permutation] = (
                torch.argsort(
                    torch.rand((n, count), dtype=self.dtype, device=self.device), dim=1
                ).to(self.dtype)
                + 1
            )
            self.lower[permutation] = 1
            self.upper[permutation] = count
        return self.evaluation(dec)

    def evaluation(self, dec) -> Population:
        """Repair and evaluate a batch of decision vectors."""
        dec = self._prepare_dec(dec)
        if dec.shape[0] == 0:
            return self._empty_population()
        obj = self.cal_obj(dec)
        con = self.cal_con(dec)
        return self._pack_population(dec, obj, con)

    def evaluate(self, dec) -> Population:
        """Evaluate decisions using the public batch-oriented problem interface."""
        return self.evaluation(dec)

    def cal_obj(self, x: torch.Tensor) -> torch.Tensor:
        """Return an ``(N, M)`` objective tensor for an ``(N, D)`` batch.

        A problem may instead override :meth:`evaluation` when objectives and
        constraints must be computed together.
        """
        raise RuntimeError(
            f"{type(self).__name__} must implement cal_obj() or override evaluation()"
        )

    def cal_con(self, x: torch.Tensor) -> torch.Tensor:
        """Compute inequality constraints for a decision batch."""
        return torch.zeros((x.shape[0], 1), dtype=x.dtype, device=x.device)

    def cal_dec(self, x: torch.Tensor) -> torch.Tensor:
        """Apply optional problem-specific repair to a decision batch.

        The generic evaluator has already enforced bounds and discrete-variable
        rounding before this hook is called. Subclasses may override the method
        for additional vectorized repair while preserving the ``(N, D)`` shape.
        """
        return x

    def get_optimum(self, n: int = 1000):
        """Return reference objective vectors for metric computation."""
        if self.M > 1:
            return torch.ones((1, self.M), dtype=self.dtype, device=self.device)
        return torch.zeros((1, 1), dtype=self.dtype, device=self.device)

    def get_pf(self):
        """Return a drawable Pareto-front representation when available."""
        return None

    @property
    def pf(self):
        """Lazily return and cache the drawable Pareto front."""
        if self._pf is None:
            self._pf = self.get_pf()
        return self._pf

    @property
    def optimum(self):
        """Lazily return and cache objective-space reference points."""
        if self._optimum is None:
            self._optimum = self.get_optimum(10000)
        return self._optimum

    def estimate_feasible_obj(self, n: int = 5000):
        """Estimate feasible objective samples for two-objective diagnostics."""
        if self.M != 2:
            return None
        if self._feasible_obj is not None:
            return self._feasible_obj
        try:
            pro = copy.deepcopy(self)
            pro.FE = 0
            dec = (
                torch.rand((n, pro.D), dtype=pro.dtype, device=pro.device) * (pro.upper - pro.lower)
                + pro.lower
            )
            pop = pro.evaluation(dec)
            feas = torch.all(pop.cons <= 0, dim=1)
            self._feasible_obj = pop.objs[feas] if torch.any(feas) else None
        except Exception:
            self._feasible_obj = None
        return self._feasible_obj

    def _empty_population(self) -> Population:
        """Return an empty population with the configured column dimensions."""
        cdim = 1 if self._con_dim is None else int(self._con_dim)
        return Population(
            torch.empty((0, self.D), dtype=self.dtype, device=self.device),
            torch.empty((0, self.M), dtype=self.dtype, device=self.device),
            torch.empty((0, cdim), dtype=self.dtype, device=self.device),
        )

    def _prepare_dec(self, dec) -> torch.Tensor:
        """Normalize, validate, bound, and repair a decision batch."""
        dec = to_tensor(dec, device=self.device, dtype=self.dtype)
        if dec.ndim == 1:
            dec = dec.unsqueeze(0)
        if dec.ndim != 2 or dec.shape[1] != self.D:
            raise ValueError(
                f"decisions must have shape (N, D={self.D}), got {tuple(dec.shape)}"
            )
        dec = torch.minimum(torch.maximum(dec, self.lower), self.upper)
        if any(self.encoding_counts[1:4]):
            dec[:, self.integer_like_mask] = round_half_away_from_zero(
                dec[:, self.integer_like_mask]
            )
        repaired = to_tensor(
            self.cal_dec(dec),
            device=self.device,
            dtype=self.dtype,
        )
        if repaired.ndim == 1:
            repaired = repaired.unsqueeze(0)
        if repaired.shape != dec.shape:
            raise ValueError(
                "cal_dec must preserve the decision batch shape "
                f"{tuple(dec.shape)}, got {tuple(repaired.shape)}"
            )
        return repaired

    def _pack_population(self, dec: torch.Tensor, obj, con) -> Population:
        """Validate evaluation outputs and assemble a population."""
        obj = to_tensor(obj, device=dec.device, dtype=dec.dtype)
        con = to_tensor(con, device=dec.device, dtype=dec.dtype)
        if obj.ndim == 1:
            obj = obj.unsqueeze(1)
        if con.ndim == 1:
            con = con.unsqueeze(1)
        if obj.ndim != 2 or obj.shape != (dec.shape[0], self.M):
            raise ValueError(
                "cal_obj must return shape "
                f"(N={dec.shape[0]}, M={self.M}), got {tuple(obj.shape)}"
            )
        if con.ndim != 2 or con.shape[0] != dec.shape[0]:
            raise ValueError(
                "cal_con must return a two-dimensional tensor with "
                f"N={dec.shape[0]} rows, got {tuple(con.shape)}"
            )
        if self._con_dim is not None and con.shape[1] != self._con_dim:
            raise ValueError(
                "cal_con must keep a stable constraint dimension across batches: "
                f"expected {self._con_dim}, got {con.shape[1]}"
            )
        self._con_dim = int(con.shape[1])
        self.FE += dec.shape[0]
        return Population(dec, obj, con)

    def _bound_tensor(self, x, default_value: float) -> torch.Tensor:
        """Expand a scalar bound or validate a per-variable bound vector."""
        x_t = to_tensor(x, device=self.device, dtype=self.dtype)
        if x_t.numel() == 0:
            return torch.full((self.D,), float(default_value), dtype=self.dtype, device=self.device)
        x_t = x_t.reshape(-1)
        if x_t.numel() == 1:
            return x_t.expand(self.D).clone()
        if x_t.numel() != self.D:
            raise ValueError(f"bound must contain 1 or D={self.D} values")
        return x_t


class Algorithm(ABC):
    """Base class for evolutionary algorithms.

    Custom algorithms implement :meth:`_solve` and retain the evolutionary
    loop inside that method. Calling :meth:`not_terminated` records history,
    dispatches callbacks, and enforces the function-evaluation budget.
    """

    name = "Algorithm"
    history_requires_snapshot = False

    def __init__(self, save: int = -10):
        """Initialize history storage.

        The absolute value of ``save`` selects the maximum number of evenly
        distributed history slots. Zero and one both retain the latest slot.
        """
        self.save = int(save)
        self.result: list[tuple[int, Population]] = []
        self.metric_cache: dict[str, torch.Tensor] = {}

    def solve(
        self,
        problem: Problem,
        output: Callable[["Algorithm", Problem], None] | None = None,
        should_stop: Callable[[], bool] | None = None,
        should_pause: Callable[[], bool] | None = None,
        *,
        device: torch.device | str | None = None,
        cpu_threads: int | None = None,
        seed: int | None = None,
    ) -> "Algorithm":
        """Run the algorithm on a problem until its evaluation budget is spent.

        Args:
            problem: Vectorized optimization problem.
            output: Callback invoked after each recorded generation.
            should_stop: Optional cooperative cancellation predicate.
            should_pause: Optional cooperative pause predicate.
            device: Explicit execution device. Environment selection is used
                when omitted.
            cpu_threads: Intra-op CPU thread count for this process.
            seed: Optional seed for Python, NumPy, PyTorch, and the accelerator.

        Returns:
            This algorithm instance after execution.
        """
        self._prepare_run(problem, device=device, cpu_threads=cpu_threads, seed=seed)
        return self._run_prepared(problem, output, should_stop, should_pause)

    def _prepare_run(
        self,
        problem: Problem,
        *,
        device: torch.device | str | None,
        cpu_threads: int | None,
        seed: int | None,
    ) -> torch.device:
        """Configure process-global execution state before a run."""
        selected_device = select_device(None if device is None else str(device))
        configure_torch(selected_device, cpu_threads=cpu_threads)
        if seed is not None:
            seed_everything(seed, selected_device)
        problem.device = selected_device
        problem.FE = 0
        self.result = []
        self.metric_cache = {}
        return selected_device

    def _run_prepared(
        self,
        problem: Problem,
        output: Callable[["Algorithm", Problem], None] | None,
        should_stop: Callable[[], bool] | None,
        should_pause: Callable[[], bool] | None,
    ) -> "Algorithm":
        """Execute a run after device, seed, and mutable state are prepared."""
        self._solve(problem, output, should_stop, should_pause)
        return self

    @abstractmethod
    def _solve(self, problem: Problem, output, should_stop, should_pause) -> None:
        """Execute the vectorized evolutionary loop."""

    def not_terminated(
        self,
        problem: Problem,
        population: Population,
        output: Callable[["Algorithm", Problem], None] | None,
        should_stop: Callable[[], bool] | None,
        should_pause: Callable[[], bool] | None,
    ) -> bool:
        """Record progress, invoke controls, and test the evaluation budget."""
        num = max(1, abs(self.save))
        index = max(
            1,
            min(
                min(num, len(self.result) + 1),
                int(math.ceil(num * problem.FE / max(problem.max_fe, 1))),
            ),
        )
        if self.history_requires_snapshot:
            pop_record = Population(
                population.decs.clone(), population.objs.clone(), population.cons.clone()
            )
        else:
            pop_record = population
        record = (problem.FE, pop_record)
        if index <= len(self.result):
            self.result[index - 1] = record
        else:
            self.result.append(record)
        if output is not None:
            output(self, problem)
        if should_stop is not None and should_stop():
            return False
        if should_pause is not None:
            while should_pause():
                if should_stop is not None and should_stop():
                    return False
                time.sleep(0.05)
        return problem.FE < problem.max_fe

    @property
    def final_population(self) -> Population:
        """Return the last recorded population.

        Raises:
            RuntimeError: If the algorithm has not recorded any generation.
        """
        if not self.result:
            raise RuntimeError("The algorithm has not produced a population")
        return self.result[-1][1]


def non_dominated_mask(obj: torch.Tensor) -> torch.Tensor:
    """Return a mask selecting nondominated rows of an objective tensor."""
    if obj.ndim != 2 or obj.shape[1] == 0:
        raise ValueError("obj must have shape (N, M) with M >= 1")
    if obj.shape[0] == 0:
        return torch.empty(0, dtype=torch.bool, device=obj.device)
    less_equal = obj[:, None, :] <= obj[None, :, :]
    less = obj[:, None, :] < obj[None, :, :]
    dominates = torch.all(less_equal, dim=2) & torch.any(less, dim=2)
    return ~torch.any(dominates, dim=0)
