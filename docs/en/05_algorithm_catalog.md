# Algorithm Directory and Selection

## Built-in Algorithms Overview

The release includes 18 built-in algorithms. The registered name is used in `create_algorithm`, while the Python class is used for direct imports.

| Category | Registered Name | Python Class | Primary Application |
| --- | --- | --- | --- |
| General | `NSGAII` | `NSGAII` | Non-dominated sorting and crowding distance |
| General | `MOEAD` | `MOEAD` | Decomposition-based multi-objective optimization |
| General | `SPEA2` | `SPEA2` | Strength fitness assignment and $k$-NN truncation |
| Constrained | `BiCo` | `BiCo` | Bi-population co-evolutionary constrained search |
| Constrained | `CMOEAD` | `CMOEAD` | Constrained decomposition-based optimization |
| Constrained | `CCMO` | `CCMO` | Co-evolutionary constrained multi-objective optimization |
| Constrained | `CMEGL` | `CMEGL` | Global and local guidance integration |
| Constrained | `CMOEACD` | `CMOEACD` | Decomposition and constraint dominance |
| Constrained | `CMOEAMS` | `CMOEAMS` | Multi-stage constraint handling |
| Constrained | `EMCMO` | `EMCMO` | Enhanced environmental selection for constraints |
| Multimodal | `HREA` | `HREA` | Joint decision- and objective-space diversity |
| Multimodal | `CMMO` | `CMMO` | Collaborative multimodal multi-objective search |
| Multimodal | `CoMMEA` | `CoMMEA` | Multi-population cooperative multimodal optimization |
| Sparse | `SparseEA` | `SparseEA` | Sparse decision variable mask evolution |
| Sparse | `SparseEA2` | `SparseEA2` | Enhanced sparse evolutionary search |
| Sparse | `TS-SparseEA` | `TSSparseEA` | Two-stage sparse evolutionary optimization |
| Sparse/Large-scale | `LMEA` | `LMEA` | Large-scale variable clustering and optimization |
| Sparse/Generative | `MOEA-PSL` | `MOEAPSL` | Pareto set learning-assisted search |

Direct import:

```python
from ascendmoea.algorithms import CMMO, MOEAPSL, NSGAII

```

Registry instantiation:

```python
from ascendmoea import create_algorithm

algorithm = create_algorithm("MOEA-PSL", save=20)

```

## Algorithm Selection Guidelines

### Selection by Problem Characteristics

* **Unconstrained / Zero Constraints**: Establish performance baselines using `NSGAII`, `MOEAD`, or `SPEA2`.
* **Constrained / Narrow Feasible Regions**: Use constrained algorithms and report feasible ratio trajectories.
* **Multimodal (Multiple Pareto-optimal solutions per objective vector)**: Use multimodal algorithms and evaluate using IGDX or PSP metrics.
* **Sparse / High-Dimensional (Few active non-zero variables)**: Use sparse optimization algorithms and track variable mask densities.

### Selection by Computational Structure

NPUs achieve optimal efficiency on large-scale matrix operations, pairwise distance computations, and batched objective evaluations. If an algorithm operates on minimal population sizes or relies heavily on data-dependent Python branching, dispatch overhead may surpass vectorized execution gains. Hardware throughput and solution quality should be evaluated concurrently.

## The `save` Parameter

All algorithms inherit `Algorithm(save=...)`. The value `abs(save)` defines the maximum number of uniformly spaced generational snapshot slots retained in `result`. Both `save=0` and `save=1` preserve only the final population state; they do not retain every single generation:

* `save=1`: Retains only the terminal state.
* `save=20` or `save=-20`: Retains at most 20 evenly spaced generation snapshots across the evaluation budget.
* To capture full generational populations: Use `HistoryMonitor` or a custom streaming monitor.

Do not rely on the internal `result` slots when exact, full-history trajectories are required.

## Compatibility Checklist

Ensure each algorithm-problem configuration satisfies the following:

1. Problem encoding matches the representation supported by algorithm operators.
2. Objective count $M$ meets reference vector and environmental selection constraints.
3. Population size $N$ is sufficient to construct required reference vectors or mating partitions.
4. Evaluation budget `max_fe` covers initialization plus at least one complete offspring batch.
5. Constrained algorithms receive consistently shaped `[N, C]` constraint tensors.

Quick sanity checks can use smaller budgets. For rigorous performance benchmarking, configure realistic budgets (e.g., $N=100$, $\text{max\_fe}=10,000$ or higher) evaluated over multiple random seeds.

## Component Inspection and Configuration-Driven Execution

```python
from ascendmoea import get_algorithm, list_algorithms

for name in list_algorithms():
    cls = get_algorithm(name)
    print(name, cls.__module__, cls.__name__)

```
