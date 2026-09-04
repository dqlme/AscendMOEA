# Core Objects and Workflow API

## Package Identifiers and Registry Catalog Objects

```python
from ascendmoea import ALGORITHMS, PROBLEMS, __version__

```

`__version__` represents the version string of the installed package. `ALGORITHMS` and `PROBLEMS` are instances of `ComponentRegistry` for algorithms and problems, respectively. While they can be queried via mapping operations, extension modules should preferably use the registration functions described later in this section to benefit from collision detection and decorator ergonomics.

## `Population`

```python
Population(decs: Tensor, objs: Tensor, cons: Tensor)

```

All three fields must be 2D tensors with identical row dimensions residing on the same device. `decs` and `objs` must contain at least one column; feasibility follows the convention $\text{cons} \le 0$. Non-tensor inputs are converted to tensors, and objective and constraint tensors are automatically placed onto the device of the decision tensor.

| Member | Return Type | Description |
| --- | --- | --- |
| `decs` | `[N, D] Tensor` | Decision variable matrix
| `objs` | `[N, M] Tensor` | Objective minimization matrix
| `cons` | `[N, C] Tensor` | Inequality constraint matrix
| `feasible` | `[N] bool Tensor` | Boolean mask indicating rows where all constraints are $\le 0$<br> |
| `best()` | `Population` | Feasible non-dominated subset; returns the single best solution for single-objective problems
| `clone()` | `Population` | Returns a deep copy of the three underlying tensors
| `detach()` | `Population` | Detaches tensors from the autograd computational graph
| `to(device, dtype=None)` | `Population` | Migrates the entire container to the target device and real precision type
| `population[index]` | `Population` | Slices the population while preserving 2D batch semantics
| `len(population)` | `int` | Number of individuals in the population

If no feasible solutions exist, `best()` returns an empty `Population` while preserving column dimensionalities. Inconsistent tensor ranks or mismatched row counts raise a `ValueError` during initialization.

## `Problem`

```python
Problem(n=100, m=2, d=10, max_fe=10_000)

```

`Problem` is an abstract base class and cannot be instantiated directly. Subclasses must implement `setting()`, and either implement `cal_obj()` or override the full `evaluation()` routine. Parameter specifications:

| Parameter | Description |
| --- | --- |
| `n` | Requested population size, stored as `N`<br> |
| `m` | Number of objectives, stored as `M`; pass `None` to resolve via `setting()`<br> |
| `d` | Decision space dimensionality, stored as `D`; pass `None` to resolve via `setting()`<br> |
| `max_fe` | Maximum budget for objective function evaluations

Primary instance attributes initialized after construction include `N`, `M`, `D`, `max_fe`, `FE`, `device`, `dtype`, `lower`, `upper`, and `encoding`. Column values in `encoding` denote variable types: 1 for Real, 2 for Integer, 3 for Categorical Label, 4 for Binary, and 5 for Permutation.

### Evaluation Methods

```python
problem.initialization(n=None) -> Population
problem.evaluate(decisions) -> Population
problem.evaluation(decisions) -> Population
problem.cal_dec(decisions) -> Tensor
problem.cal_obj(decisions) -> Tensor
problem.cal_con(decisions) -> Tensor

```

`evaluate()` is the public alias for `evaluation()`. Inputs accept shapes `[D]` or `[N, D]` and are processed as batches. The execution pipeline casts inputs to the problem's device and dtype, applies boundary clamping and discrete encoding repairs, and subsequently calls `cal_dec`, `cal_obj`, and `cal_con`. Objective evaluations must return shape `[N, M]`, and constraint evaluations must return shape `[N, C]`, with constant column dimension $C$ across the problem lifecycle. Upon successful evaluation, `FE` increments by the evaluated batch size; invalid output shapes raise a `ValueError` without incrementing `FE`.

### Reference Data

| Property / Method | Description |
| --- | --- |
| `optimum` | Lazily evaluated and cached objective-space reference points
| `pf` | Lazily evaluated and cached drawable Pareto front geometry; may be `None`<br> |
| `get_optimum(n=1000)` | Extensible entry point for generating objective-space reference points
| `get_pf()` | Extensible entry point for generating drawable Pareto front structures
| `estimate_feasible_obj(n=5000)` | Samples estimated feasible objective vectors for bi-objective problems

Built-in multimodal problems additionally define `ps` and `reference_point`, which are not standard attributes across arbitrary `Problem` implementations.

## `Algorithm`

```python
Algorithm(save=-10)

```

Subclasses implement `_solve(problem, output, should_stop, should_pause)`. The attribute `abs(save)` configures the maximum number of uniformly spaced generational snapshots retained within the algorithm's internal `result`; both 0 and 1 retain only the most recent generation. Each record is stored as a tuple `(evaluations, population)`. To record an exhaustive generational history, attach a `HistoryMonitor`.

```python
algorithm.solve(
    problem,
    output=None,
    should_stop=None,
    should_pause=None,
    *,
    device=None,
    cpu_threads=None,
    seed=None,
) -> Algorithm

```

`solve()` configures hardware devices, threading limits, and PRNG states, resets execution history and `problem.FE`, and executes optimization. It returns the active algorithm instance. `final_population` returns the terminal population snapshot, raising a `RuntimeError` if called prior to execution. Direct calls to `solve()` do not track execution runtimes; use `optimize()` or `Workflow` for synchronized wall-clock timing.

## Standard Workflow API

### `RunConfig`

```python
RunConfig(
    device=None,
    seed=None,
    cpu_threads=None,
    synchronize_timing=True,
)

```

A frozen configuration dataclass storing execution parameters for a single run. Setting `device=None` defers to environment configuration. When running on CPU without an explicit thread count, the runtime falls back to `ASCENDMOEA_CPU_THREADS`, defaulting to single-threaded execution. `synchronize_timing=True` enforces hardware queue synchronization immediately before and after execution timing.

### `Workflow`

```python
Workflow(algorithm, problem, monitors=None, config=None)
workflow.run(output=None, should_stop=None, should_pause=None) -> OptimizationResult

```

The `monitors` argument accepts a single `Monitor` instance or a sequence of monitors. The `output` callback is invoked at every generational boundary. Returning `True` from `should_stop` triggers cooperative termination; returning `True` from `should_pause` suspends execution while continuing to poll termination conditions. Runtime exceptions are routed to all attached monitor `on_error` hooks prior to being propagated.

### `optimize`

```python
optimize(
    algorithm,
    problem,
    *,
    device=None,
    seed=None,
    cpu_threads=None,
    monitors=None,
    synchronize_timing=True,
) -> OptimizationResult

```

A convenience entry point that constructs a `RunConfig` and `Workflow` to immediately execute an optimization run.

### `OptimizationResult`

| Member | Type | Description |
| --- | --- | --- |
| `algorithm` | `Algorithm` | The completed algorithm instance
| `problem` | `Problem` | The completed problem instance
| `device` | `torch.device` | The active compute device
| `elapsed_seconds` | `float` | Synchronized wall-clock runtime in seconds
| `population` | `Population` | Final recorded population snapshot
| `evaluations` | `int` | Total consumed evaluations (`problem.FE`)

## Monitors

The `Monitor` base class defines four lifecycle hooks with default no-op implementations:

```python
on_start(algorithm, problem) -> None
on_generation(algorithm, problem) -> None
on_finish(result) -> None
on_error(algorithm, problem, error) -> None

```

`HistoryMonitor(copy_to_cpu=True)` clones and stores the population at each generation. `records` contains a list of `GenerationRecord` objects, each holding `evaluations` and `population` fields. For large-scale experiments, use streaming monitors to prevent memory usage from growing with the number of generations.

## Component Registry

```python
ComponentRegistry(kind, components=())
registry.register(component=None, *, name=None, replace=False)
registry.create(name, **kwargs)
registry.names() -> tuple[str, ...]

```

Registries implement the `MutableMapping[str, type]` interface. Stored values must be class types. Name collisions raise a `KeyError` by default and require explicit `replace=True` to overwrite. Dedicated registration hooks are provided for algorithms and problems:

```python
register_algorithm(component=None, *, name=None, replace=False)
register_problem(component=None, *, name=None, replace=False)
get_algorithm(name) -> type[Algorithm]
get_problem(name) -> type[Problem]
create_algorithm(name, **kwargs) -> Algorithm
create_problem(name, **kwargs) -> Problem
list_algorithms() -> tuple[str, ...]
list_problems() -> tuple[str, ...]

```

`get_*` raises a `KeyError` for unknown identifiers; `create_*` forwards keyword arguments directly to class constructors; `list_*` returns lexicographically sorted names for configuration validation and logging.

## Device Management API

| Function Signature | Description |
| --- | --- |
| `select_device(requested=None)` | Resolves and validates `cpu`, `cuda:k`, `npu:k`, or `auto` device targets
| `npu_is_available()` | Loads the device backend on demand and verifies NPU hardware availability
| `configure_torch(device, cpu_threads=None)` | Sets floating-point dtype, CPU threads, or the active accelerator context
| `seed_everything(seed, device=None)` | Sets deterministic seeds across Python, NumPy, PyTorch, and accelerator backends
| `synchronize(device)` | Blocks host execution until all queued operations on the device complete

Supported environment variables include `ASCENDMOEA_DEVICE`, `ASCENDMOEA_NPU_INDEX`, `ASCENDMOEA_CPU_THREADS`, and `ASCENDMOEA_DTYPE`. NPU execution defaults to `float32`; requesting `float64` issues a warning and falls back to `float32`.
