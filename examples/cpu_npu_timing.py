"""Compare synchronized CPU and Ascend NPU execution times."""

from ascendmoea import npu_is_available, optimize
from ascendmoea.algorithms import NSGAII
from ascendmoea.problems import DTLZ2


def run(device: str) -> float:
    """Run one reproducible NSGA-II benchmark and return synchronized time."""
    result = optimize(
        NSGAII(save=1),
        DTLZ2(n=1000, m=3, max_fe=100_000),
        device=device,
        cpu_threads=8 if device == "cpu" else None,
        seed=42,
    )
    return result.elapsed_seconds


def main() -> None:
    """Report CPU timing and NPU timing when an Ascend device is available."""
    print(f"cpu_seconds={run('cpu'):.6f}")
    if npu_is_available():
        print(f"npu_seconds={run('npu:0'):.6f}")


if __name__ == "__main__":
    main()
