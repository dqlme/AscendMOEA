# Operators, Metrics, Experiments, and Plotting API

## Public Population Operators

All operators are imported from `ascendmoea.operators`. Input tensors must reside on the same compute device; outputs preserve the input tensor device and dtype unless explicitly specified.

### Dominance and Sorting

```python
domination_matrix(pop_obj: Tensor) -> BoolTensor
domination_count(pop_obj: Tensor) -> Tensor
non_dominated_mask(obj: Tensor) -> BoolTensor
nd_sort(pop_obj: Tensor, pop_con: Tensor | None = None,
        n_sort: int | float = inf) -> tuple[Tensor, int]

```

* `domination_matrix` accepts `[N, M]` and returns an `[N, N]` boolean matrix where $(i, j)$ is `True` if solution $i$ dominates $j$ under minimization.

* `domination_count` computes the number of solutions dominating each row.

* `non_dominated_mask` evaluates unconstrained dominance across objective matrices.

* `nd_sort` returns assigned front ranks and the maximum front index. Constraint violations are identified where `pop_con > 0`; `n_sort` bounds the number of individuals requiring definitive rank assignment.

Invalid matrix ranks, empty objective columns, row count mismatches between objectives and constraints, or negative `n_sort` values raise a `ValueError`.

### Mating Selection

```python
tournament_selection(k: int, n: int, *fitness: Tensor) -> LongTensor
roulette_selection(n: int, fitness: Tensor) -> LongTensor

```

`tournament_selection` returns $n$ indices, evaluating input fitness tensors lexicographically in the provided order (lower values represent higher priority/superior fitness). Requires $k \ge 1$ and at least one non-empty fitness tensor of matching length.

`roulette_selection` treats inputs as minimization fitness values, applying non-negative shifts and drawing samples proportionally to reciprocal probabilities. It does not accept direct maximization probability weights, and $n$ must be non-negative.

### Diversity and Distance Metrics

```python
crowding_distance(pop_obj: Tensor, front_no: Tensor) -> Tensor
pairwise_dist(a: Tensor, b: Tensor) -> Tensor
sequential_distance_truncation(distance: Tensor, delete_count: int) -> BoolTensor

```

`crowding_distance` computes crowding distances within each finite front, assigning infinity ($\infty$) to boundary extremes. `pairwise_dist` accepts `[A, D]` and `[B, D]`, returning the Euclidean distance matrix `[A, B]` using matrix-multiplication paths with numerical error correction for large inputs.

`sequential_distance_truncation` accepts a square distance matrix and returns a boolean deletion mask. On NPU targets, distance matrices are constructed on-device while sequential truncation iterations run on CPU to avoid excessive small-kernel launch overhead. When `AscendMOEA[accelerator]` is installed, truncation workloads exceeding internal size thresholds execute via compiled accelerator implementations.

```python
sequential_truncation_diagnostics() -> dict[str, object]
reset_sequential_truncation_diagnostics() -> None
warmup_sequential_truncation_accelerator(dtype: torch.dtype | None = None) -> dict[str, object]

```

The environment variable `ASCENDMOEA_COMPILED_TRUNCATION` accepts `auto`, `off`, or `required`.

### Variation Operators

```python
operator_ga(problem, parent_dec, pro_c=1.0, dis_c=20.0,
            pro_m=1.0, dis_m=20.0, half=False) -> Tensor
operator_ga_encoded(problem, parent_dec, pro_c=1.0, dis_c=20.0,
                    pro_m=1.0, dis_m=20.0, half=False) -> Tensor
operator_de(problem, p1, p2, p3, cr=1.0, f=0.5,
            pro_m=1.0, dis_m=20.0) -> Tensor

```

`operator_ga` applies crossover and mutation based on `problem.encoding` across real, integer, label, binary, and permutation columns. Parent rows are partitioned into halves for mating (the trailing row of an odd-sized batch is dropped); `half=True` outputs one offspring per parent pair. Requires at least two parent rows with column dimensionality matching `problem.D`.

`operator_ga_encoded` provides a standalone entry point with explicit encoding awareness. `genetic_operator` is exported as an alias for `operator_ga`.

`operator_de` computes $p_1 + F \times (p_2 - p_3)$ combined with binomial crossover and polynomial mutation. Parent matrices `p1`, `p2`, and `p3` must share identical shapes with column count equal to `problem.D`. `differential_evolution_operator` is exported as an alias.

### Sampling and Population Operations

```python
uniform_point(n: int, m: int, device=None) -> tuple[Tensor, int]
merge_pop(*populations: Population) -> Population

```

`uniform_point` requires $n \ge 1$ and $m \ge 2$, returning simplex-lattice reference vectors alongside the actual generated count, which may differ from the requested $n$. `merge_pop` concatenates at least one `Population` container along the individual dimension.

## Quality Indicators

```python
igd(pop, optimum) -> float
gd(pop, optimum) -> float
igdp(pop, optimum) -> float
delta_p(pop, optimum) -> float
dm(pop, optimum) -> float
cpf(pop, optimum) -> float
hv(pop, optimum) -> float
igdx(pop, pos) -> float
coverage_rate(pop, pos) -> float
psp(pop, pos) -> float
feasible_rate(pop, _optimum=None) -> float

```

Objective-space metrics evaluate feasible non-dominated solutions via `population.best()`; decision-space metrics evaluate the complete array `population.decs` to retain diversity coverage across multimodal landscapes. Reference datasets accept 1D coordinate vectors or 2D matrices; missing, empty, or dimension-mismatched reference inputs evaluate to `NaN`. Objective-space indicators evaluate to `NaN` if no feasible solutions exist.

| Registered Metric | Function | Optimization Goal | Reference Space |
| --- | --- | --- | --- |
| `IGD` | `igd` | Minimization | Objective
| `HV` | `hv` | Maximization | Objective
| `GD` | `gd` | Minimization | Objective
| `IGDp` | `igdp` | Minimization | Objective
| `DeltaP` | `delta_p` | Minimization | Objective
| `DM` | `dm` | Maximization | Objective
| `CPF` | `cpf` | Maximization | Objective
| `IGDX` | `igdx` | Minimization | Decision
| `CR` | `coverage_rate` | Maximization | Decision
| `PSP` | `psp` | Minimization | Decision
| `Feasible_rate` | `feasible_rate` | Maximization | None
| `runtime` | `None` | Minimization | Workflow Execution

The Pure Spread Distance metric is defined as $\text{PSP} = \text{IGDX} / \text{CR}$, evaluating to positive infinity ($+\infty$) when coverage rate evaluates to zero. Hypervolume uses exact slicing methods for 2-objective and 3-objective tasks, while scaling to Monte Carlo sampling ($10^6$ device samples) for formulations with 4 or more objectives. Because high-dimensional HV relies on Monte Carlo estimation, fix random seeds across comparative experiments.

The `METRICS` dictionary maps registered indicator names to tuples of `(callable, "min" | "max")`, where `runtime` maps to `None`.

## Statistical Experimentation Framework

Requires `AscendMOEA[experiment]`. Exports the following interfaces:

```python
RunRecord(algorithm: str, problem: str, run: int, metrics: dict[str, float])

run_single(alg, pro, output_callback=None, should_stop_cb=None, *,
           device=None, seed=None, cpu_threads=None)
    -> tuple[dict[str, float], Algorithm]

run_experiment(alg_factory, pro_factory, n_runs, *,
               device=None, base_seed=None, cpu_threads=None)
    -> list[RunRecord]

records_to_frame(records) -> pandas.DataFrame
summarize(df, metric, style="mean_std") -> pandas.DataFrame
significance_tests(df, metric, test, baseline_alg=None) -> list[dict]

```

`run_single` executes an optimization run using the synchronized workflow and computes all compatible quality indicators. Objective metrics evaluate against `pro.optimum`; IGDX, CR, and PSP evaluate against `pro.ps`, returning `NaN` if reference sets are unavailable or incompatible.

`run_experiment` invokes factories per repetition to instantiate fresh algorithm and problem objects. If `base_seed` is provided, iteration $r$ uses `base_seed + r - 1`; setting `n_runs < 1` raises a `ValueError`. Repetitions run sequentially; parallel scaling across seeds must be managed by external process schedulers.

The `summarize` parameter `style` accepts `mean`, `mean_std`, or `median_iqr`. The `significance_tests` parameter `test` accepts `friedman`, `wilcoxon`, or `ranksum` (with aliases `signrank` and `ranksums`). Friedman tests require paired evaluations across at least three algorithms; pairwise tests require specifying `baseline_alg`. If observations are insufficient, an empty list is returned.

## Plotting API

Requires `AscendMOEA[plot]`. All visualization routines return a `plotly.graph_objects.Figure` with detached tensors converted to CPU `float64` precision. Default typography uses Times New Roman.

```python
draw_obj(obj, pf=None, title="Population (objectives)") -> Figure
draw_problem_obj(problem, pop_obj, title="Population (objectives)",
                 include_pf=True) -> Figure
update_population_trace(fig, pop_obj) -> Figure
draw_dec(dec, title="Population (variables)") -> Figure
draw_metric_curve(fes, values, name) -> Figure

```

* `draw_obj` accepts tensors of shape `[N, 2]` or `[N, 3]`.

* `draw_problem_obj` renders reference Pareto front geometry using `problem.pf` and `problem.optimum`.

* `update_population_trace` performs in-place trace updates on the first figure trace named `Population`.

* `draw_dec` generates scatter plots for 1D, 2D, and 3D decisions, falling back to parallel coordinate plots for higher dimensions.

* `draw_metric_curve` requires matching array lengths across evaluation counters (`FE`) and metric values.
