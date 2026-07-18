"""Composable execution workflow with synchronized timing and monitoring."""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import torch

from .core import Algorithm, Population, Problem
from .device import select_device, synchronize


@dataclass(frozen=True)
class RunConfig:
    """Define reproducible execution settings for one optimization run.

    Args:
        device: Device string or PyTorch device. ``None`` uses environment-based
            automatic selection.
        seed: Optional random seed shared by all supported random generators.
        cpu_threads: Intra-op CPU thread count for this process.
        synchronize_timing: Synchronize accelerator queues around wall timing.
    """

    device: torch.device | str | None = None
    seed: int | None = None
    cpu_threads: int | None = None
    synchronize_timing: bool = True


@dataclass(frozen=True)
class GenerationRecord:
    """Represent one monitored population snapshot."""

    evaluations: int
    population: Population


@dataclass(frozen=True)
class OptimizationResult:
    """Contain the completed run, final population, and synchronized runtime."""

    algorithm: Algorithm
    problem: Problem
    device: torch.device
    elapsed_seconds: float

    @property
    def population(self) -> Population:
        """Return the final recorded population."""
        return self.algorithm.final_population

    @property
    def evaluations(self) -> int:
        """Return the number of completed function evaluations."""
        return int(self.problem.FE)


class Monitor:
    """Base class for workflow lifecycle monitors."""

    def on_start(self, algorithm: Algorithm, problem: Problem) -> None:
        """Handle workflow initialization before timed execution."""
        return None

    def on_generation(self, algorithm: Algorithm, problem: Problem) -> None:
        """Handle a newly recorded generation."""
        return None

    def on_finish(self, result: OptimizationResult) -> None:
        """Handle successful workflow completion."""
        return None

    def on_error(
        self,
        algorithm: Algorithm,
        problem: Problem,
        error: BaseException,
    ) -> None:
        """Handle an exception before it is propagated to the caller."""
        return None


class HistoryMonitor(Monitor):
    """Collect complete generation snapshots independently of history slots.

    Args:
        copy_to_cpu: Move snapshots to CPU to avoid retaining accelerator memory.
    """

    def __init__(self, copy_to_cpu: bool = True) -> None:
        """Create an empty in-memory generation history."""
        self.copy_to_cpu = bool(copy_to_cpu)
        self.records: list[GenerationRecord] = []

    def on_start(self, algorithm: Algorithm, problem: Problem) -> None:
        """Clear records before a new workflow run."""
        self.records.clear()

    def on_generation(self, algorithm: Algorithm, problem: Problem) -> None:
        """Clone the current population into the monitor history."""
        population = algorithm.final_population.detach().clone()
        if self.copy_to_cpu:
            population = population.to("cpu")
        self.records.append(GenerationRecord(int(problem.FE), population))


class Workflow:
    """Bind an algorithm, a problem, execution settings, and monitors.

    Args:
        algorithm: Algorithm instance to execute.
        problem: Problem instance to optimize.
        monitors: Optional monitor or sequence of monitors.
        config: Reproducible run configuration.
    """

    def __init__(
        self,
        algorithm: Algorithm,
        problem: Problem,
        monitors: Monitor | Sequence[Monitor] | None = None,
        config: RunConfig | None = None,
    ) -> None:
        """Create a reusable workflow from one algorithm/problem pair."""
        self.algorithm = algorithm
        self.problem = problem
        if monitors is None:
            self.monitors: tuple[Monitor, ...] = ()
        elif isinstance(monitors, Monitor):
            self.monitors = (monitors,)
        else:
            self.monitors = tuple(monitors)
        self.config = config or RunConfig()

    def run(
        self,
        output: Callable[[Algorithm, Problem], None] | None = None,
        should_stop: Callable[[], bool] | None = None,
        should_pause: Callable[[], bool] | None = None,
    ) -> OptimizationResult:
        """Execute the configured optimization and return its result.

        Accelerator timing excludes outstanding work from earlier operations
        and includes all work submitted by the algorithm.
        """
        device = select_device(None if self.config.device is None else str(self.config.device))
        self.algorithm._prepare_run(
            self.problem,
            device=device,
            cpu_threads=self.config.cpu_threads,
            seed=self.config.seed,
        )

        for monitor in self.monitors:
            monitor.on_start(self.algorithm, self.problem)

        def generation_callback(algorithm: Algorithm, problem: Problem) -> None:
            """Forward a generation boundary to monitors and the user callback."""
            for monitor in self.monitors:
                monitor.on_generation(algorithm, problem)
            if output is not None:
                output(algorithm, problem)

        if self.config.synchronize_timing:
            synchronize(device)
        started = time.perf_counter()
        try:
            self.algorithm._run_prepared(
                self.problem,
                generation_callback,
                should_stop,
                should_pause,
            )
            if self.config.synchronize_timing:
                synchronize(device)
            elapsed = time.perf_counter() - started
        except BaseException as error:
            for monitor in self.monitors:
                monitor.on_error(self.algorithm, self.problem, error)
            raise

        result = OptimizationResult(
            algorithm=self.algorithm,
            problem=self.problem,
            device=device,
            elapsed_seconds=elapsed,
        )
        for monitor in self.monitors:
            monitor.on_finish(result)
        return result


def optimize(
    algorithm: Algorithm,
    problem: Problem,
    *,
    device: torch.device | str | None = None,
    seed: int | None = None,
    cpu_threads: int | None = None,
    monitors: Monitor | Sequence[Monitor] | None = None,
    synchronize_timing: bool = True,
) -> OptimizationResult:
    """Run one algorithm/problem pair through the standard workflow."""
    config = RunConfig(
        device=device,
        seed=seed,
        cpu_threads=cpu_threads,
        synchronize_timing=synchronize_timing,
    )
    return Workflow(algorithm, problem, monitors=monitors, config=config).run()


__all__ = [
    "GenerationRecord",
    "HistoryMonitor",
    "Monitor",
    "OptimizationResult",
    "RunConfig",
    "Workflow",
    "optimize",
]
