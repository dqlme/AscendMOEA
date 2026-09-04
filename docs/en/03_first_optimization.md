# First Optimization Task

## Running NSGA-II on DTLZ2

```python
from ascendmoea import HistoryMonitor, optimize
from ascendmoea.algorithms import NSGAII
from ascendmoea.problems import DTLZ2

history = HistoryMonitor(copy_to_cpu=True)
result = optimize(
    algorithm=NSGAII(save=20),
    problem=DTLZ2(n=100, m=3, max_fe=10_000),
    device="cpu",
    cpu_threads=8,
    seed=42,
    monitors=history,
)

print("evaluations", result.evaluations)
print("elapsed_seconds", result.elapsed_seconds)
print("objective_shape", tuple(result.population.objs.shape))
print("saved_generations", len(history.records))

```

Object instantiation is separated from workflow orchestration: the algorithm maintains its internal state, the problem defines bounds, evaluation budgets, and reference data, and `optimize` handles device placement, seeding, synchronized timing, and monitor lifecycles.

## Switching to Ascend NPU

```python
result = optimize(
    NSGAII(save=20),
    DTLZ2(n=1000, m=3, max_fe=100_000),
    device="npu:0",
    seed=42,
)
print(result.elapsed_seconds)

```

Because the NPU relies on an asynchronous execution queue, wrapping optimization runs in unsynchronized `time.perf_counter()` calls will produce inaccurate benchmarks. The standard workflow synchronizes timing by default; set `synchronize_timing=False` only when isolating dispatch overhead.

## Creating Components by Name

The component registry enables string-based instantiation for configuration-driven pipelines:

```python
from ascendmoea import create_algorithm, create_problem, optimize

algorithm = create_algorithm("MOEAD", save=10)
problem = create_problem("ZDT1", n=200, max_fe=20_000)
result = optimize(algorithm, problem, device="cpu", cpu_threads=8, seed=1)

```

Query available component names:

```python
from ascendmoea import list_algorithms, list_problems

print(list_algorithms())
print(list_problems())

```

## Accessing Population Data

Primary `Population` attributes are 2D tensors:

| Field | Tensor Shape | Description |
| --- | --- | --- |
| `decs` | `[N, D]` | Decision variables |
| `objs` | `[N, M]` | Objective function values |
| `cons` | `[N, C]` | Constraint violations (feasible when $\le 0$) |

`Population` instances support slicing, cloning, graph detaching, and device transfers:

```python
front = result.population[:20].clone().to("cpu")
print(front.decs, front.objs, front.cons)

```

Retrieve the feasible, non-dominated subset using `population.best()`.

## Evaluating Performance Metrics

```python
from ascendmoea.metrics import hv, igd, igdp

reference_front = result.problem.optimum
print("HV", hv(result.population, reference_front))
print("IGD", igd(result.population, reference_front))
print("IGD+", igdp(result.population, reference_front))

```

For multimodal optimization, compute `igdx` using analytical or generated reference Pareto sets. For formal benchmarks, the sampling method and size of reference sets must be documented and kept constant across trials.

## Termination, Pausing, and Output Callbacks

Custom callbacks can be passed directly via `Workflow.run`:

```python
from pathlib import Path
from ascendmoea import RunConfig, Workflow

stop_file = Path("STOP")

def report(algorithm, problem):
    print(problem.FE, algorithm.final_population.objs.shape)

workflow = Workflow(
    NSGAII(),
    DTLZ2(n=100, m=3, max_fe=10_000),
    config=RunConfig(device="cpu", seed=9, cpu_threads=8),
)
result = workflow.run(output=report, should_stop=stop_file.exists)

```

The `should_stop` condition is evaluated at generation boundaries. For fault-tolerant long runs, orchestrate runs using external process managers (such as systemd or SLURM) and persist results atomically per repetition.
