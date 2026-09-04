# Tensorized Operators Manual

## Import Protocol

Public operators must be imported exclusively from `ascendmoea.operators`:

```python
from ascendmoea.operators import nd_sort, operator_ga, tournament_selection

```

Do not import operators from private `utils` or internal algorithm modules. Internal modules may be refactored across minor versions, whereas the public namespace serves as the stable public API surface.

## Dominance Relationships and Sorting

### `domination_matrix(pop_obj)`

Accepts an `[N, M]` minimization objective tensor and returns an `[N, N]` boolean dominance matrix, where element $(i, j)$ is `True` if solution $i$ dominates solution $j$. Memory complexity scales as $\mathcal{O}(N^2)$; stress-testing large population sizes requires monitoring peak device allocation.

### `domination_count(pop_obj)`

Computes the number of individuals dominating each candidate solution. Used for strength fitness assignment and rapid candidate filtering.

### `nd_sort(pop_obj, pop_con=None, n_sort=inf)`

Computes non-dominated sorting and returns `(front_no, max_front)`. Constraint violations adhere to $g(x) \le 0$; infeasible solutions are ranked according to their aggregated constraint violation magnitude. The `n_sort` parameter bounds the number of individuals requiring definitive front assignments; pass `n_sort=N` during environmental selection to eliminate unnecessary sorting computations on lower fronts.

## Mating Selection

### `tournament_selection(k, n, *fitness)`

Executes a $k$-way tournament selection and returns $n$ indices. Multiple input fitness tensors are evaluated lexicographically in the provided order (lower values represent higher priority/superior fitness). Standard NSGA-II mating selection:

```python
parents = tournament_selection(2, problem.N, front_no, -crowding)

```

### `roulette_selection(n, fitness)`

Treats input values as minimization fitness scores, applying non-negative shifts and drawing $n$ indices proportionally to reciprocal probabilities. All fitness entries must be non-empty, finite numbers. It does not accept direct maximization probability weights.

## Diversity and Distance Metrics

### `crowding_distance(pop_obj, front_no)`

Calculates crowding distances independently within each non-domination front, returning an `[N]` tensor. Boundary extremes are assigned infinity ($\infty$). The underlying implementation uses vectorized sorting and scatter operations to maintain stable tie-breaking semantics.

### `pairwise_dist(a, b)`

Computes the pairwise Euclidean distance matrix of shape `[A, B]` between coordinate tensors `[A, D]` and `[B, D]`. When the product $A \times B$ is large, the resulting matrix constitutes the primary memory allocation; memory ceiling breaches trigger explicit out-of-memory errors.

### `sequential_distance_truncation(distance, delete_count)`

Executes SPEA2-style sequential distance truncation, returning a boolean retention mask or indexing tensor. Verify accelerator compilation paths using the diagnostics API:

```python
from ascendmoea.operators import (
    reset_sequential_truncation_diagnostics,
    sequential_truncation_diagnostics,
    warmup_sequential_truncation_accelerator,
)

reset_sequential_truncation_diagnostics()
warmup_sequential_truncation_accelerator()
print(sequential_truncation_diagnostics())

```

Diagnostic metadata serves strictly to verify runtime execution paths and must not be fed back into algorithm fitness logic.

## Variation Operators

### `operator_ga(problem, parent_dec, pro_c=1, dis_c=20, pro_m=1, dis_m=20, half=False)`

Applies corresponding genetic crossover and mutation operators based on `problem.encoding` across real, integer, label, binary, and permutation variables. The input argument is a raw decision tensor, not a `Population` container. Setting `half=True` generates a single offspring per parent pair instead of two.

`genetic_operator` is exported as an exact semantic alias.

### `operator_ga_encoded(...)`

Provides an explicit encoding-aware interface for algorithmic pipelines requiring variable-specific genetic parameterization. Validate parameter signatures to ensure matching tensor bounds, encodings, and device placements.

### `operator_de(problem, p1, p2, p3, cr=1, f=0.5, pro_m=1, dis_m=20)`

Executes batched differential evolution mutation with polynomial mutation. The row dimensions of `p1`, `p2`, and `p3` must match.
`differential_evolution_operator` is exported as an exact semantic alias.

## Sampling and Population Operations

### `uniform_point(n, m, device=None)`

Generates a uniformly distributed reference vector simplex, returning `(points, actual_n)`. Algorithms must consume the returned `actual_n`, as simplex combinatorial constraints may prevent matching the requested $n$ exactly.

### `merge_pop(*populations)`

Concatenates multiple `Population` instances along the individual batch dimension. Column counts, device allocations, and tensor dtypes across all underlying containers must be identical.

### `non_dominated_mask(obj)`

Returns a boolean mask identifying unconstrained non-dominated solutions within an objective matrix. For constrained formulations, filter for feasible candidates first or invoke `nd_sort(obj, con)`.

## Device and Precision Guidelines

* All operator tensor arguments must reside on the same compute device.

* Avoid unparameterized allocations such as `torch.zeros` or `torch.arange` inside operators without passing target `device` and `dtype` parameters.

* Use stable sorting routines to guarantee deterministic tie-breaking behavior across repeated runs.

* Warm up operator execution with matching tensor shapes and synchronize devices prior to wall-clock benchmarking on Ascend NPUs.

* When validating CPU versus NPU equivalence, apply dual absolute/relative tolerance checks ($\text{atol} + \text{rtol}$) and cross-verify discrete sorting ranks, masks, and indices separately.

* For $\mathcal{O}(N^2)$ operators, simultaneously track wall-clock latency, peak host memory, and peak device HBM allocation.

## Standalone Operator Example

```python
from ascendmoea.operators import crowding_distance, nd_sort, operator_ga
from ascendmoea.problems import DTLZ2

problem = DTLZ2(n=100, m=3, max_fe=10_000)
population = problem.initialization()
front_no, _ = nd_sort(population.objs, population.cons, problem.N)
crowding = crowding_distance(population.objs, front_no)
offspring_dec = operator_ga(problem, population.decs)
offspring = problem.evaluate(offspring_dec)

```
