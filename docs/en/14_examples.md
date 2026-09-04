# Runnable Examples

## Example 1: Reproducible CPU8 Baseline

```python
from ascendmoea import HistoryMonitor, optimize
from ascendmoea.algorithms import NSGAII
from ascendmoea.metrics import hv, igd
from ascendmoea.problems import DTLZ2

problem = DTLZ2(n=100, m=3, max_fe=10_000)
monitor = HistoryMonitor(copy_to_cpu=True)
result = optimize(
    NSGAII(save=1),
    problem,
    device="cpu",
    cpu_threads=8,
    seed=1001,
    monitors=monitor,
)
reference = problem.optimum
print("seconds", result.elapsed_seconds)
print("HV", hv(result.population, reference))
print("IGD", igd(result.population, reference))

```

## Example 2: Switching the Same Task to NPU

```python
problem = DTLZ2(n=1000, m=3, max_fe=100_000)
result = optimize(
    NSGAII(save=1),
    problem,
    device="npu:0",
    seed=1001,
)
print(result.elapsed_seconds)

```

Reported wall-clock time includes start/stop device synchronizations. Rigorous benchmarking requires shape warm-up runs, fixed CPU8 threading, exclusive NPU card allocation, and 20 independent seed runs.

## Example 3: Constrained Optimization and Feasible Ratio

```python
from ascendmoea.algorithms import CMOEAD
from ascendmoea.metrics import feasible_rate, igd
from ascendmoea.problems import LIRCMOP5

problem = LIRCMOP5(n=100, max_fe=10_000)
result = optimize(CMOEAD(), problem, device="cpu", cpu_threads=8, seed=1002)
print("feasible_rate", feasible_rate(result.population))
print("igd", igd(result.population, problem.optimum))

```

## Example 4: Multimodal Decision-Space Evaluation

```python
from ascendmoea.algorithms import CMMO
from ascendmoea.metrics import igd, igdx
from ascendmoea.problems import MMF1

problem = MMF1(n=100, max_fe=10_000)
result = optimize(CMMO(), problem, device="cpu", cpu_threads=8, seed=1003)
reference_ps = problem.ps
print("IGD", igd(result.population, problem.optimum))
print("IGDX", igdx(result.population, reference_ps))

```

Built-in multimodal problems include reference Pareto sets in the distribution. Custom problems must generate or load an equivalent decision-space reference set and document its version metadata alongside experimental results.

## Example 5: Sparse Large-Scale Optimization

```python
from ascendmoea.algorithms import SparseEA
from ascendmoea.problems import SMOP1

problem = SMOP1(n=200, d=1000, max_fe=100_000)
result = optimize(SparseEA(), problem, device="npu:0", seed=1004)
active_ratio = (result.population.decs != 0).to(result.population.decs.dtype).mean()
print("active_ratio", active_ratio.item())

```

Extracting `.item()` after completion avoids blocking asynchronous execution queues within generational loops.

## Example 6: Using Public Operators Standalone

```python
from ascendmoea.operators import (
    crowding_distance,
    merge_pop,
    nd_sort,
    operator_ga,
    tournament_selection,
)
from ascendmoea.problems import ZDT1

problem = ZDT1(n=100, max_fe=10_000)
population = problem.initialization()
front, _ = nd_sort(population.objs, population.cons, problem.N)
crowding = crowding_distance(population.objs, front)
parents = tournament_selection(2, problem.N, front, -crowding)
offspring = problem.evaluate(operator_ga(problem, population.decs[parents]))
combined = merge_pop(population, offspring)
print(combined.objs.shape)

```

## Example 7: Configuration-Driven Batch Execution

```python
from ascendmoea import create_algorithm, create_problem, optimize

def run(config):
    return optimize(
        create_algorithm(config["algorithm"], save=1),
        create_problem(
            config["problem"],
            n=config["n"],
            max_fe=config["max_fe"],
        ),
        device=config["device"],
        cpu_threads=config.get("cpu_threads"),
        seed=config["seed"],
    )

```

Instantiate fresh algorithm and problem objects per worker process. Save experimental artifacts in isolated directories with explicit completion markers as described in Chapter 10.

## Example 8: Custom Population Monitor

```python
from ascendmoea import Monitor

class MetricMonitor(Monitor):
    def __init__(self):
        self.rows = []

    def on_generation(self, algorithm, problem):
        population = algorithm.final_population.detach().to("cpu")
        self.rows.append(
            {
                "fe": problem.FE,
                "feasible": float((population.cons <= 0).all(dim=1).float().mean()),
            }
        )

```

For large-scale production experiments, stream data to disk and deallocate population copies within callbacks rather than unbounded in-memory list appending.
