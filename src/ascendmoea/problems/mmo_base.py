"""Vectorized implementation of the mmo_base problem."""

from __future__ import annotations

import importlib
import os
import threading
from functools import lru_cache
from typing import Callable

import torch

from ..core import Problem
from ..utils import pairwise_dist
from ..vendor.mmo_reference.functions.Basic.load_functions import load_functions

MMO_ID_TO_NAME = {
    1: "MMF1",
    2: "MMF1_e",
    3: "MMF1_z",
    4: "MMF2",
    5: "MMF3",
    6: "MMF4",
    7: "MMF5",
    8: "MMF6",
    9: "MMF7",
    10: "MMF8",
    11: "MMF9",
    12: "MMF10",
    13: "MMF11",
    14: "MMF12",
    15: "MMF13",
    16: "MMF14",
    17: "MMF14_a",
    18: "MMF15",
    19: "MMF15_a",
    20: "SYM_PART_simple",
    21: "SYM_PART_rotated",
    22: "Omni_test",
    23: "IDMPM2T1",
    24: "IDMPM2T2",
    25: "IDMPM2T3",
    26: "IDMPM2T4",
    27: "IDMPM3T1",
    28: "IDMPM3T2",
    29: "IDMPM3T3",
    30: "IDMPM3T4",
    31: "IDMPM4T1",
    32: "IDMPM4T2",
    33: "IDMPM4T3",
    34: "IDMPM4T4",
    35: "IDMPM2T1_e",
    36: "IDMPM2T2_e",
    37: "IDMPM2T3_e",
    38: "IDMPM2T4_e",
    39: "IDMPM3T1_e",
    40: "IDMPM3T2_e",
    41: "IDMPM3T3_e",
    42: "IDMPM3T4_e",
    43: "MMMOP1A",
    44: "MMMOP1B",
    45: "MMMOP2A",
    46: "MMMOP2B",
    47: "MMMOP3A",
    48: "MMMOP3B",
    49: "MMMOP3C",
    50: "MMMOP3D",
    51: "MMMOP4A",
    52: "MMMOP4B",
    53: "MMMOP4C",
    54: "MMMOP4D",
    55: "MMMOP5A",
    56: "MMMOP5B",
    57: "MMMOP5C",
    58: "MMMOP5D",
    59: "MMMOP6A",
    60: "MMMOP6B",
    61: "MMMOP6C",
    62: "MMMOP6D",
    63: "poly_2",
    64: "poly_3",
    65: "poly_4",
    66: "poly_5",
    67: "poly_6",
    68: "poly_8",
    69: "poly_10",
    70: "poly_12",
    71: "map",
    72: "application",
}

MMO_NAME_TO_ID = {v: k for k, v in MMO_ID_TO_NAME.items()}
MMO_PROBLEM_NAMES = list(MMO_NAME_TO_ID)


def _real_world_data_dir() -> str:
    """Return the packaged multimodal real-world data directory."""
    return os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "vendor", "multimodal_real_world")
    )


@lru_cache(maxsize=None)
def _load_mmo_spec(problem_id: int):
    """Load one immutable multimodal problem specification."""
    with _FORMULA_EVALUATION_LOCK:
        previous_dtype = torch.get_default_dtype()
        try:
            # Avoid permanently quantizing CPU bounds/reference points to fp32.
            torch.set_default_dtype(torch.float64)
            return load_functions(problem_id)
        finally:
            torch.set_default_dtype(previous_dtype)


@lru_cache(maxsize=None)
def _load_app_mat(filename: str):
    """Load one packaged multimodal real-world MAT file."""
    import scipy.io

    path = os.path.join(_real_world_data_dir(), filename)
    return scipy.io.loadmat(path)


class _TorchFactoryProxy:
    """Inject the current tensor's device and dtype into reference formulas."""

    def __init__(self, like: torch.Tensor):
        """Initialize the suite problem and apply its fixed configuration."""
        self.like = like

    def __getattr__(self, name):
        """Delegate an unresolved attribute lookup."""
        return getattr(torch, name)

    def _kwargs(self, kwargs):
        """Fill omitted factory device and dtype from the evaluated batch."""
        kwargs = dict(kwargs)
        kwargs.setdefault("device", self.like.device)
        kwargs.setdefault("dtype", self.like.dtype)
        return kwargs

    def tensor(self, data, *args, **kwargs):
        """Create a tensor colocated with the evaluated batch."""
        return torch.tensor(data, *args, **self._kwargs(kwargs))

    def zeros(self, *args, **kwargs):
        """Create a zero tensor colocated with the evaluated batch."""
        return torch.zeros(*args, **self._kwargs(kwargs))

    def ones(self, *args, **kwargs):
        """Create a one tensor colocated with the evaluated batch."""
        return torch.ones(*args, **self._kwargs(kwargs))

    def empty(self, *args, **kwargs):
        """Create an uninitialized tensor colocated with the evaluated batch."""
        return torch.empty(*args, **self._kwargs(kwargs))

    def full(self, *args, **kwargs):
        """Create a filled tensor colocated with the evaluated batch."""
        return torch.full(*args, **self._kwargs(kwargs))

    def arange(self, *args, **kwargs):
        """Create an integer range on the evaluated batch device."""
        kwargs = dict(kwargs)
        kwargs.setdefault("device", self.like.device)
        return torch.arange(*args, **kwargs)

    def linspace(self, *args, **kwargs):
        """Create an evenly spaced tensor colocated with the evaluated batch."""
        return torch.linspace(*args, **self._kwargs(kwargs))


_FORMULA_EVALUATION_LOCK = threading.RLock()


def _call_with_torch_proxy(
    fn: Callable[[torch.Tensor], torch.Tensor], x: torch.Tensor
) -> torch.Tensor:
    """Evaluate a legacy formula with batch-aware tensor factories."""
    module = importlib.import_module(fn.__module__)
    with _FORMULA_EVALUATION_LOCK:
        original_torch = getattr(module, "torch", torch)
        module.torch = _TorchFactoryProxy(x)
        try:
            y = fn(x)
        finally:
            module.torch = original_torch
    return torch.as_tensor(y, dtype=x.dtype, device=x.device)


def _rlp(x: torch.Tensor) -> torch.Tensor:
    """Compute rlp."""
    data = _load_app_mat("RLP_data.mat")
    points_a = torch.as_tensor(data["A"][:, :2], dtype=x.dtype, device=x.device)
    points_b = torch.as_tensor(data["B"][:, :2], dtype=x.dtype, device=x.device)
    points_c = torch.as_tensor(data["C"][:, :2], dtype=x.dtype, device=x.device)
    x2 = x[:, :2]
    return torch.column_stack(
        [
            pairwise_dist(x2, points_a).min(dim=1).values,
            pairwise_dist(x2, points_b).min(dim=1).values,
            pairwise_dist(x2, points_c).min(dim=1).values,
        ]
    )


def _ueclp(x: torch.Tensor) -> torch.Tensor:
    """Compute ueclp."""
    points1 = torch.tensor([[-2.0, -1.0], [-1.0, -1.0]], dtype=x.dtype, device=x.device)
    points2 = torch.tensor([[2.0, 1.0], [1.0, 1.0]], dtype=x.dtype, device=x.device)
    points3 = torch.tensor([[-2.0, 1.0], [-1.0, 1.0]], dtype=x.dtype, device=x.device)
    points4 = torch.tensor([[2.0, -1.0], [1.0, -1.0]], dtype=x.dtype, device=x.device)
    y = torch.empty((x.shape[0], 2), dtype=x.dtype, device=x.device)
    idx1 = (x[:, 0] < 0) & (x[:, 1] < 0)
    idx2 = (x[:, 0] > 0) & (x[:, 1] > 0)
    idx3 = (x[:, 0] < 0) & (x[:, 1] > 0)
    idx4 = ~(idx1 | idx2 | idx3)
    if torch.any(idx1):
        y[idx1] = pairwise_dist(x[idx1, :2], points1)
    if torch.any(idx2):
        y[idx2] = pairwise_dist(x[idx2, :2], points2)
    if torch.any(idx3):
        y[idx3] = pairwise_dist(x[idx3, :2], points3)
    if torch.any(idx4):
        y[idx4] = pairwise_dist(x[idx4, :2], points4)
    y = torch.where((x[:, [0]] >= 0), y + 1.0, y)
    return y


class MMOProblem(Problem):
    """Define the MMOProblem benchmark with batched tensor evaluation."""

    def __init__(self, problem_id: int | str, n=100, max_fe=10000):
        """Initialize the suite problem and apply its fixed configuration."""
        if isinstance(problem_id, str):
            problem_id = MMO_NAME_TO_ID[problem_id]
        self.problem_id = int(problem_id)
        self._function = None
        self._ps = None
        self._pf_data = None
        self._refpoint = None
        super().__init__(n=n, m=None, d=None, max_fe=max_fe)

    def setting(self):
        """Configure problem dimensions, bounds, encoding, and static data."""
        if self.problem_id in MMO_ID_TO_NAME:
            function, function_name, n_var, n_obj, xl, xu, refpoint, ps, pf = _load_mmo_spec(
                self.problem_id
            )
            self._function = function
            self.name = function_name
            self.M = int(n_obj)
            self.D = int(n_var)
            self.lower = xl.reshape(-1).to(device=self.device, dtype=self.dtype)
            self.upper = xu.reshape(-1).to(device=self.device, dtype=self.dtype)
            self._refpoint = (
                None
                if refpoint is None
                else refpoint.reshape(-1).to(device=self.device, dtype=self.dtype)
            )
            self._ps = None if ps is None else ps.to(device=self.device, dtype=self.dtype)
            self._pf_data = None if pf is None else pf.to(device=self.device, dtype=self.dtype)
            return
        if self.problem_id == 73:
            self.name = "RLP"
            self.M = 3
            self.D = 2
            self.lower = torch.tensor([0.0, 0.0], dtype=self.dtype, device=self.device)
            self.upper = torch.tensor([23.0, 23.0], dtype=self.dtype, device=self.device)
            self._function = _rlp
            data = _load_app_mat("RLP_data.mat")
            self._pf_data = torch.as_tensor(data["PF"], dtype=self.dtype, device=self.device)
            self._ps = torch.as_tensor(data["PS"], dtype=self.dtype, device=self.device)
            return
        if self.problem_id == 74:
            self.name = "UECLP"
            self.M = 2
            self.D = 2
            self.lower = torch.tensor([-5.0, -5.0], dtype=self.dtype, device=self.device)
            self.upper = torch.tensor([5.0, 5.0], dtype=self.dtype, device=self.device)
            self._function = _ueclp
            data = _load_app_mat("UECLP.mat")
            self._pf_data = torch.as_tensor(data["PF"], dtype=self.dtype, device=self.device)
            self._ps = torch.as_tensor(data["PS"], dtype=self.dtype, device=self.device)
            return
        raise KeyError(f"Unknown MMO problem id: {self.problem_id}")

    def cal_obj(self, x: torch.Tensor) -> torch.Tensor:
        """Evaluate objective values for a batch of decision vectors."""
        if self.problem_id in MMO_ID_TO_NAME:
            return _call_with_torch_proxy(self._function, x)
        return self._function(x)

    def cal_con(self, x: torch.Tensor) -> torch.Tensor:
        """Evaluate inequality constraints for a batch of decision vectors."""
        return torch.zeros((x.shape[0], 1), dtype=x.dtype, device=x.device)

    def get_optimum(self, n: int = 1000):
        """Generate objective-space reference points for metrics."""
        if self._pf_data is not None:
            # Keep the complete fixed set because rows may group distinct fronts.
            return self._pf_data
        return super().get_optimum(n)

    @property
    def reference_point(self):
        """Return the recommended objective-space reference point, if defined."""
        return self._refpoint

    def get_pf(self):
        """Generate a drawable Pareto-front representation."""
        return self._pf_data

    @property
    def ps(self):
        """Return the packaged decision-space Pareto set, if available."""
        return self._ps


class MMOProblemFactory:
    """Construct one registered multimodal problem from its stable identifier."""

    def __init__(self, problem_id: int):
        """Initialize the suite problem and apply its fixed configuration."""
        self.problem_id = int(problem_id)
        if problem_id in MMO_ID_TO_NAME:
            self.name = MMO_ID_TO_NAME[problem_id]
        else:
            self.name = {73: "RLP", 74: "UECLP"}[problem_id]

    def __call__(self, **kwargs):
        """Create the configured multimodal problem."""
        return MMOProblem(self.problem_id, **kwargs)


MMO_CLASSES = {name: MMOProblemFactory(pid) for pid, name in MMO_ID_TO_NAME.items()}
MMO_CLASSES["RLP"] = MMOProblemFactory(73)
MMO_CLASSES["UECLP"] = MMOProblemFactory(74)
