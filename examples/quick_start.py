"""Run NSGA-II on DTLZ2 through the standard synchronized workflow."""

from ascendmoea import optimize
from ascendmoea.algorithms import NSGAII
from ascendmoea.problems import DTLZ2


def main() -> None:
    """Execute and report a reproducible CPU example."""
    algorithm = NSGAII(save=20)
    problem = DTLZ2(n=100, m=3, max_fe=10_000)
    result = optimize(
        algorithm,
        problem,
        device="cpu",
        cpu_threads=8,
        seed=42,
    )
    print(f"device={result.device}")
    print(f"evaluations={result.evaluations}")
    print(f"population={len(result.population)}")
    print(f"elapsed_seconds={result.elapsed_seconds:.6f}")


if __name__ == "__main__":
    main()
