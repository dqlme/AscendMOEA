"""Validate the packaged API, operators, problems, and algorithms on one device."""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from importlib import metadata, resources
from pathlib import Path

import torch

from ascendmoea import (
    Algorithm,
    Problem,
    RunConfig,
    Workflow,
    __version__,
    configure_torch,
    create_algorithm,
    create_problem,
    register_algorithm,
    register_problem,
    seed_everything,
    select_device,
    synchronize,
)
from ascendmoea import operators as public_operators
from ascendmoea.algorithms import ALGORITHMS
from ascendmoea.metrics import (
    METRICS,
    coverage_rate,
    cpf,
    delta_p,
    dm,
    feasible_rate,
    gd,
    hv,
    igd,
    igdp,
    igdx,
    psp,
)
from ascendmoea.operators import (
    crowding_distance,
    differential_evolution_operator,
    domination_count,
    domination_matrix,
    genetic_operator,
    merge_pop,
    nd_sort,
    non_dominated_mask,
    operator_de,
    operator_ga,
    operator_ga_encoded,
    pairwise_dist,
    reset_sequential_truncation_diagnostics,
    roulette_selection,
    sequential_distance_truncation,
    sequential_truncation_diagnostics,
    tournament_selection,
    uniform_point,
    warmup_sequential_truncation_accelerator,
)
from ascendmoea.problems import DTLZ2, PROBLEMS

CONSTRAINED = {
    "BiCo",
    "CMOEAD",
    "CCMO",
    "CMEGL",
    "CMOEACD",
    "GANCMOEACD",
    "MTCCMOEACD",
    "CMOEAMS",
    "EMCMO",
}
MULTIMODAL = {"HREA", "CMMO", "CoMMEA"}
SPARSE = {"SparseEA", "SparseEA2", "TS-SparseEA", "LMEA", "MOEA-PSL"}
BUILTIN_ALGORITHM_NAMES = ALGORITHMS.names()
BUILTIN_PROBLEM_NAMES = PROBLEMS.names()


@dataclass
class CheckResult:
    """Represent one release validation check."""

    category: str
    name: str
    status: str
    elapsed_seconds: float
    details: dict[str, object]


@register_problem(name="ReleaseCustomProblem", replace=True)
class ReleaseCustomProblem(Problem):
    """Small vectorized problem used to validate third-party extension points."""

    name = "ReleaseCustomProblem"

    def __init__(self, n: int = 20, max_fe: int = 80) -> None:
        """Initialize the custom validation problem."""
        super().__init__(n=n, m=2, d=6, max_fe=max_fe)

    def setting(self) -> None:
        """Set continuous bounds for every decision variable."""
        self.lower = torch.full((self.D,), -2.0, dtype=self.dtype, device=self.device)
        self.upper = torch.full((self.D,), 2.0, dtype=self.dtype, device=self.device)

    def cal_obj(self, x: torch.Tensor) -> torch.Tensor:
        """Evaluate two shifted quadratic objectives as one batch."""
        return torch.column_stack((x.square().sum(1), (x - 1.0).square().sum(1)))


@register_algorithm(name="ReleaseCustomAlgorithm", replace=True)
class ReleaseCustomAlgorithm(Algorithm):
    """Minimal vectorized algorithm used to validate custom algorithm loading."""

    name = "ReleaseCustomAlgorithm"

    def _solve(self, problem: Problem, output, should_stop, should_pause) -> None:
        """Generate random batches and retain nondominated candidates."""
        population = problem.initialization()
        while self.not_terminated(
            problem,
            population,
            output,
            should_stop,
            should_pause,
        ):
            combined = merge_pop(population, problem.initialization(problem.N))
            front_no, last = nd_sort(combined.objs, combined.cons, problem.N)
            crowding = crowding_distance(combined.objs, front_no)
            selected = front_no < last
            remaining = problem.N - int(selected.sum().item())
            if remaining > 0:
                final_front = torch.where(front_no == last)[0]
                order = torch.argsort(crowding[final_front], descending=True, stable=True)
                selected[final_front[order[:remaining]]] = True
            population = combined[selected]


def timed(device: torch.device, function):
    """Measure a callable with synchronization around accelerator work."""
    synchronize(device)
    started = time.perf_counter()
    value = function()
    synchronize(device)
    return value, time.perf_counter() - started


def assert_finite(name: str, tensor: torch.Tensor) -> None:
    """Raise a descriptive error when a tensor contains non-finite values."""
    if not torch.all(torch.isfinite(tensor)):
        raise RuntimeError(f"{name} produced non-finite values")


def require(condition: bool, message: str) -> None:
    """Raise a release-validation error when a required invariant is false."""
    if not condition:
        raise RuntimeError(message)


def installed_version(distribution: str) -> str | None:
    """Return an installed distribution version without requiring it."""
    try:
        return metadata.version(distribution)
    except metadata.PackageNotFoundError:
        return None


def count_package_resources(node, suffix: str) -> int:
    """Recursively count package resources with a selected suffix."""
    total = 0
    for child in node.iterdir():
        if child.is_dir():
            total += count_package_resources(child, suffix)
        elif child.name.lower().endswith(suffix.lower()):
            total += 1
    return total


def validate_operators(device: torch.device) -> CheckResult:
    """Exercise every operator exported by the public operator namespace."""
    started = time.perf_counter()
    problem = DTLZ2(n=20, m=3, max_fe=200)
    problem.device = device
    population = problem.initialization()
    matrix = domination_matrix(population.objs)
    counts = domination_count(population.objs)
    fronts, _ = nd_sort(population.objs, population.cons, problem.N)
    crowding = crowding_distance(population.objs, fronts)
    parents = tournament_selection(2, problem.N, fronts, -crowding)
    roulette = roulette_selection(problem.N, counts.to(problem.dtype) + 1)
    offspring_dec = operator_ga(problem, population.decs[parents])
    encoded_dec = operator_ga_encoded(problem, population.decs[parents])
    de_dec = operator_de(
        problem,
        population.decs,
        population.decs.roll(1, 0),
        population.decs.roll(2, 0),
    )
    offspring = problem.evaluate(offspring_dec)
    combined = merge_pop(population, offspring)
    distance = pairwise_dist(combined.objs, combined.objs)
    reset_sequential_truncation_diagnostics()
    deleted = sequential_distance_truncation(distance, 3)
    references, count = uniform_point(20, 3, device=device)
    nondominated = non_dominated_mask(combined.objs)
    diagnostics = sequential_truncation_diagnostics()
    accelerator = warmup_sequential_truncation_accelerator(problem.dtype)

    require(matrix.shape == (20, 20), "domination_matrix returned an invalid shape")
    require(counts.shape == (20,), "domination_count returned an invalid shape")
    require(
        parents.shape == (20,) and roulette.shape == (20,),
        "selection operators returned invalid shapes",
    )
    require(encoded_dec.shape == offspring_dec.shape, "encoded GA returned an invalid shape")
    require(de_dec.shape == population.decs.shape, "DE returned an invalid shape")
    require(genetic_operator is operator_ga, "genetic_operator is not the documented alias")
    require(
        differential_evolution_operator is operator_de,
        "differential_evolution_operator is not the documented alias",
    )
    require(
        deleted.shape == (40,) and int(deleted.sum().item()) == 3,
        "sequential truncation returned an invalid deletion mask",
    )
    require(references.shape == (count, 3), "uniform_point returned an invalid shape")
    require(nondominated.shape == (40,), "non_dominated_mask returned an invalid shape")
    assert_finite("genetic operator", offspring_dec)
    assert_finite("encoded genetic operator", encoded_dec)
    assert_finite("differential evolution operator", de_dec)
    assert_finite("pairwise distance", distance)
    return CheckResult(
        category="operators",
        name="public_operator_surface",
        status="ok",
        elapsed_seconds=time.perf_counter() - started,
        details={
            "operator_count": len(public_operators.__all__),
            "reference_count": count,
            "truncation_diagnostics": diagnostics,
            "accelerator": accelerator,
        },
    )


def validate_metrics(device: torch.device) -> CheckResult:
    """Exercise every public metric against objective and decision references."""
    started = time.perf_counter()
    problem = DTLZ2(n=20, m=3, max_fe=40)
    problem.device = device
    population = problem.initialization()
    objective_reference = problem.optimum
    decision_reference = population.decs.clone()
    values = {
        "IGD": igd(population, objective_reference),
        "HV": hv(population, objective_reference),
        "GD": gd(population, objective_reference),
        "IGDp": igdp(population, objective_reference),
        "DeltaP": delta_p(population, objective_reference),
        "DM": dm(population, objective_reference),
        "CPF": cpf(population, objective_reference),
        "IGDX": igdx(population, decision_reference),
        "CR": coverage_rate(population, decision_reference),
        "PSP": psp(population, decision_reference),
        "Feasible_rate": feasible_rate(population),
    }
    if set(values).difference(METRICS):
        raise RuntimeError("The metric registry is missing public metrics")
    nonfinite = {name: value for name, value in values.items() if not torch.isfinite(torch.tensor(value))}
    if nonfinite:
        raise RuntimeError(f"Metrics produced non-finite values: {nonfinite}")
    return CheckResult(
        category="metrics",
        name="public_metric_surface",
        status="ok",
        elapsed_seconds=time.perf_counter() - started,
        details={"metric_count": len(METRICS), "values": values},
    )


def validate_resources() -> CheckResult:
    """Verify that benchmark data are accessible through package resources."""
    started = time.perf_counter()
    package_root = resources.files("ascendmoea")
    mat_count = count_package_resources(package_root.joinpath("vendor"), ".mat")
    if mat_count < 1:
        raise RuntimeError("No bundled MAT resources were found")
    return CheckResult(
        category="resources",
        name="bundled_benchmark_data",
        status="ok",
        elapsed_seconds=time.perf_counter() - started,
        details={"mat_files": mat_count},
    )


def problem_kwargs(name: str) -> dict[str, int]:
    """Return lightweight construction arguments for a registered problem."""
    if name.startswith("DASCMOP"):
        return {"n": 4, "max_fe": 4}
    return {"n": 4, "max_fe": 4}


def validate_problems(device: torch.device) -> list[CheckResult]:
    """Construct and batch-evaluate every registered problem."""
    rows: list[CheckResult] = []
    for position, name in enumerate(BUILTIN_PROBLEM_NAMES):
        seed_everything(83000 + position, device)
        started = time.perf_counter()
        problem = create_problem(name, **problem_kwargs(name))
        problem.device = device
        population = problem.initialization(min(problem.N, 4))
        expected_rows = min(problem.N, 4)
        require(
            population.objs.shape == (expected_rows, problem.M),
            f"{name} returned an invalid objective shape",
        )
        require(problem.lower.shape == (problem.D,), f"{name} has invalid lower bounds")
        require(problem.upper.shape == (problem.D,), f"{name} has invalid upper bounds")
        require(problem.encoding.shape == (problem.D,), f"{name} has invalid encoding")
        require(bool(torch.all(problem.lower <= problem.upper).item()), f"{name} has reversed bounds")
        assert_finite(name + " lower bounds", problem.lower)
        assert_finite(name + " upper bounds", problem.upper)
        assert_finite(name + " objectives", population.objs)
        assert_finite(name + " constraints", population.cons)
        rows.append(
            CheckResult(
                category="problem",
                name=name,
                status="ok",
                elapsed_seconds=time.perf_counter() - started,
                details={"M": problem.M, "D": problem.D, "FE": problem.FE},
            )
        )
    return rows


def algorithm_case(name: str) -> tuple[str, dict[str, int]]:
    """Return a lightweight compatible problem for an algorithm."""
    if name in CONSTRAINED:
        return "CF1", {"n": 20, "d": 10, "max_fe": 180}
    if name in SPARSE:
        return "SMOP1", {"n": 12, "d": 30, "max_fe": 180}
    if name in MULTIMODAL:
        return "MMF1", {"n": 20, "max_fe": 160}
    return "MMF1", {"n": 20, "max_fe": 120}


def validate_algorithms(
    device: torch.device,
    cpu_threads: int | None,
    selected: set[str] | None,
) -> list[CheckResult]:
    """Run every selected built-in algorithm on a compatible problem."""
    rows: list[CheckResult] = []
    names = [name for name in BUILTIN_ALGORITHM_NAMES if selected is None or name in selected]
    for position, name in enumerate(names):
        problem_name, kwargs = algorithm_case(name)
        problem = create_problem(problem_name, **kwargs)
        algorithm = create_algorithm(name, save=1)
        result = Workflow(
            algorithm,
            problem,
            config=RunConfig(
                device=device,
                seed=71000 + position,
                cpu_threads=cpu_threads,
            ),
        ).run()
        population = result.population
        require(len(population) > 0, f"{name} returned an empty population")
        assert_finite(name + " objectives", population.objs)
        assert_finite(name + " constraints", population.cons)
        rows.append(
            CheckResult(
                category="algorithm",
                name=name,
                status="ok",
                elapsed_seconds=result.elapsed_seconds,
                details={
                    "problem": problem_name,
                    "FE": result.evaluations,
                    "population": len(population),
                },
            )
        )
    return rows


def validate_extensions(
    device: torch.device,
    cpu_threads: int | None,
) -> CheckResult:
    """Run a registry-created custom algorithm on a custom problem."""
    result = Workflow(
        create_algorithm("ReleaseCustomAlgorithm", save=1),
        create_problem("ReleaseCustomProblem"),
        config=RunConfig(device=device, seed=99001, cpu_threads=cpu_threads),
    ).run()
    require(len(result.population) == 20, "custom extension returned an invalid population size")
    assert_finite("custom extension objectives", result.population.objs)
    return CheckResult(
        category="extension",
        name="custom_algorithm_and_problem",
        status="ok",
        elapsed_seconds=result.elapsed_seconds,
        details={"FE": result.evaluations, "population": len(result.population)},
    )


def write_report(path: Path, report: dict[str, object]) -> None:
    """Write a UTF-8 JSON report atomically enough for validation logging."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> int:
    """Parse command-line options and run the complete release validation."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--cpu-threads", type=int, default=1)
    parser.add_argument("--algorithms", default="")
    parser.add_argument("--skip-problems", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    device = select_device(args.device)
    cpu_threads = args.cpu_threads if device.type == "cpu" else None
    configure_torch(device, cpu_threads=cpu_threads)
    selected = {name for name in args.algorithms.split(",") if name} or None
    unknown = set() if selected is None else selected.difference(BUILTIN_ALGORITHM_NAMES)
    if unknown:
        raise ValueError(f"Unknown algorithms: {sorted(unknown)}")

    started = time.perf_counter()
    results: list[CheckResult] = [
        validate_resources(),
        validate_operators(device),
        validate_metrics(device),
    ]
    if not args.skip_problems:
        results.extend(validate_problems(device))
    results.extend(validate_algorithms(device, cpu_threads, selected))
    results.append(validate_extensions(device, cpu_threads))
    report = {
        "schema_version": 1,
        "package": "AscendMOEA",
        "version": __version__,
        "checked_at": datetime.now(timezone.utc).astimezone().isoformat(),
        "device": str(device),
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "dependencies": {
            name: installed_version(name)
            for name in ("numpy", "scipy", "pandas", "plotly", "numba", "torch-npu")
        },
        "status": "ok",
        "elapsed_seconds": time.perf_counter() - started,
        "counts": {
            "algorithms": sum(row.category == "algorithm" for row in results),
            "problems": sum(row.category == "problem" for row in results),
            "operators": sum(row.category == "operators" for row in results),
            "metrics": sum(row.category == "metrics" for row in results),
            "resources": sum(row.category == "resources" for row in results),
            "extensions": sum(row.category == "extension" for row in results),
        },
        "results": [asdict(row) for row in results],
    }
    write_report(args.output, report)
    print(json.dumps(report["counts"], ensure_ascii=False), flush=True)
    print(f"status=ok output={args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
