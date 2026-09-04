# Public API Index

This index lists stable public modules; full signatures, parameters, return types, and exceptions are documented across four reference pages. Symbols omitted from these pages or module-level `__all__` definitions are internal implementation details and may change across versions without deprecation notices.

## Core and Execution

[Core and Workflow API](16_core_api.md) covers:

* `Population`, `Problem`, `Algorithm`

* `RunConfig`, `Workflow`, `OptimizationResult`

* `Monitor`, `HistoryMonitor`, `GenerationRecord`

* Device selection, thread configuration, random seeding, and device synchronization

* `ComponentRegistry` and factory functions for algorithms and problems

## Algorithms

[Algorithm API](17_algorithm_api.md) details all 18 built-in algorithms, covering registered identifiers, class names, constructor signatures, parameter boundaries, and target applications. Both direct imports and configuration-based factories access the same underlying registry:

```python
from ascendmoea.algorithms import NSGAII
from ascendmoea import create_algorithm

direct = NSGAII(save=10)
configured = create_algorithm("NSGAII", save=10)

```

## Problems

[Problem API](18_problem_api.md) documents all 155 registered problems, problem suite constructor parameters, variable encoding schemes, reference data properties, and direct import conventions.

## Operators, Metrics, Experiments, and Plotting

[Operators and Analytics API](19_analysis_api.md) covers:

* The 19 public operators exported in `ascendmoea.operators`

* The 11 quality indicators and the `METRICS` registry in `ascendmoea.metrics`

* Replication orchestration, aggregation, and statistical significance testing in `ascendmoea.experiment`

* The five visualization plotting functions in `ascendmoea.plotting`

## Top-Level Package Exports

The top-level `ascendmoea` namespace exports the following public symbols:

```text
ALGORITHMS, PROBLEMS, Algorithm, ComponentRegistry, GenerationRecord,
HistoryMonitor, Monitor, OptimizationResult, Population, Problem, RunConfig,
Workflow, __version__, configure_torch, create_algorithm, create_problem,
get_algorithm, get_problem, list_algorithms, list_problems, npu_is_available,
optimize, register_algorithm, register_problem, seed_everything, select_device,
synchronize

```
