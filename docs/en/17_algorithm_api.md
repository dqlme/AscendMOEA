# Built-in Algorithm API

All algorithm classes can be imported from `ascendmoea.algorithms` or instantiated via `create_algorithm(name, **kwargs)`. The `**kwargs` parameter passes base class arguments such as `save`.

The algorithm submodule exports `ALGORITHMS`, `register_algorithm`, `get_algorithm`, `create_algorithm`, and `list_algorithms`, which reference the shared global registry. Factory methods raise a `KeyError` for unregistered names, and registration attempts with duplicate identifiers are rejected by default.

## Algorithm Catalog

| Category | Registered Name | Class Name | Constructor Signature |
| --- | --- | --- | --- |
| General | `NSGAII` | `NSGAII` | `(save=-10)`<br> |
| General | `MOEAD` | `MOEAD` | `(agg_type=1, **kwargs)`<br> |
| General | `SPEA2` | `SPEA2` | `(save=-10)`<br> |
| Constrained | `BiCo` | `BiCo` | `(save=-10)`<br> |
| Constrained | `CMOEAD` | `CMOEAD` | `(save=-10)`<br> |
| Constrained | `CCMO` | `CCMO` | `(op_type=1, **kwargs)`<br> |
| Constrained | `CMEGL` | `CMEGL` | `(save=-10)`<br> |
| Constrained | `CMOEACD` | `CMOEACD` | `(e1=1, e2=1, **kwargs)`<br> |
| Constrained | `CMOEAMS` | `CMOEAMS` | `(op_type=1, lam=0.5, **kwargs)`<br> |
| Constrained | `EMCMO` | `EMCMO` | `(save=-10)`<br> |
| Multimodal | `HREA` | `HREA` | `(eps=0.3, p=0.5, **kwargs)`<br> |
| Multimodal | `CMMO` | `CMMO` | `(eta=0.2, tao=0.1, theta=0.1, **kwargs)`<br> |
| Multimodal | `CoMMEA` | `CoMMEA` | `(eps=0.2, **kwargs)`<br> |
| Sparse | `SparseEA` | `SparseEA` | `(save=-10)`<br> |
| Sparse | `SparseEA2` | `SparseEA2` | `(save=-10)`<br> |
| Sparse | `TS-SparseEA` | `TSSparseEA` | `(save=-10, r_eva=0.1, n_group=50)`<br> |
| Large-Scale | `LMEA` | `LMEA` | `(save=-10, n_sel=5, n_per=50, n_cor=5, operator_type=1)`<br> |
| Sparse/Generative | `MOEA-PSL` | `MOEAPSL` | `(save=-10)`<br> |

## General-Purpose Algorithms

### `NSGAII`

Implements environmental selection based on constraint-aware non-dominated sorting and objective-space crowding distance. Falls back to standard unconstrained selection when constraints are absent or zero. Accepts no additional hyperparameter arguments.

### `MOEAD`

Performs subproblem updates within localized weight vector neighborhoods. The `agg_type` parameter accepts:

| Value | Aggregation Function |
| --- | --- |
| 1 | Penalty-based Boundary Intersection (PBI, penalty parameter $\theta=5$)
| 2 | Weighted Tchebycheff
| 3 | Normalized Weighted Tchebycheff
| 4 | Modified Tchebycheff

Because simplex reference vector generators may adjust final population sizing, reference `problem.N` and output population shapes after execution.

### `SPEA2`

Implements strength fitness assignment, $k$-nearest neighbor density estimation, and sequential distance truncation. Large-scale truncation routines optionally dispatch to a Numba-compiled CPU accelerator path, falling back to a reference implementation when unavailable.

## Constrained Optimization Algorithms

### `BiCo`

Maintains a primary population alongside an infeasible archive, balancing candidate selection across constraint violations, objective directions, and sequential truncation.

### `CMOEAD`

Integrates constraint violation values into decomposition-based neighborhood replacement schemes, applying replacement bounds per subproblem and switching between local and global mating pools.

### `CCMO`

Co-evolves a constraint-handling primary population alongside an unconstrained auxiliary population. `op_type=1` applies genetic variation operators; `op_type=2` uses differential evolution operators; invalid option values raise a `ValueError`.

### `CMEGL`

Maintains constrained, global, and local population sets, adaptively balancing global and local information using objective distribution statistics and feasibility ratios.

### `CMOEACD`

Maintains exploration, feasibility-exploitation, and diversity archives. Parameters `e1` and `e2` configure selection modes for exploration and feasibility archives, respectively:

| Value | Environmental Selection Method |
| --- | --- |
| 1 | Sequential distance truncation
| 2 | Crowding distance truncation
| 3 | Reference vector association

Both parameters must be set to 1, 2, or 3.

### `CMOEAMS`

Transitions across optimization phases—from a joint objective-constraint search to a constraint-dominance phase—based on feasibility ratios and consumed evaluation budgets. `op_type=1` uses GA operators; `op_type=2` uses DE operators; `lam` defines the feasibility threshold parameter ($\lambda \in [0, 1]$).

### `EMCMO`

Maintains constrained and unconstrained auxiliary populations, adaptively adjusting solution migration based on the search state. Accepts no additional constructor arguments.

## Multimodal Optimization Algorithms

### `HREA`

Maintains populations and archives using hierarchical objective partitioning and decision-space neighborhood metrics. The parameter `eps` defines a non-negative decision neighborhood radius; `p` specifies the archive mating selection probability applied after half the evaluation budget is consumed ($p \in [0, 1]$).

### `CMMO`

Co-evolves a convergence-focused population and a decision-space diversity population. The parameters `eta`, `tao`, and `theta` configure the initial adaptive epsilon threshold, decay schedule, and decision-space neighborhood radius, respectively (all parameters require values in $[0, 1]$).

### `CoMMEA`

Exchanges offspring solutions between global convergence search and local decision-space exploitation sub-populations. `eps` specifies the lower bound for the adaptive local convergence radius ($\epsilon > 0$).

## Sparse and Large-Scale Algorithms

### `SparseEA`

Co-evolves decision variable values alongside boolean activation masks, using variable-level fitness scores to guide sparse crossover and mutation.

### `SparseEA2`

Extends sparse mask evolution by introducing variable-grouping mutations sorted by decision magnitudes, fixing the number of partitions to 4.

### `TSSparseEA`

Executes an initial variable-group mask search before transitioning into individual-level sparse optimization. `r_eva` specifies the proportion of evaluation budget allocated to the group search phase ($r_{\text{eva}} \in (0, 1]$); `n_group` defines the number of variable partitions ($n_{\text{group}} \in \mathbb{Z}^+$).

### `LMEA`

Applies variable clustering and correlation analysis prior to alternating optimization across distance-related and position-related variable sets. Parameters include:

| Parameter | Description |
| --- | --- |
| `n_sel` | Number of population samples selected for variable analysis
| `n_per` | Perturbation sample count per selected individual
| `n_cor` | Number of correlation evaluations per variable pair
| `operator_type` | Variation operator mode: 1 for GA, 2 for DE

The first three counting parameters require positive integers.

### `MOEA-PSL`

Uses a generative latent variable model to learn sparse Pareto set manifolds, combining model sampling with mask evolution. Accepts no additional public constructor arguments.

## Execution and Output Handling

All algorithms are executed via `Algorithm.solve()`, `Workflow`, or `optimize()`, and the final population is retrieved through `OptimizationResult.population`. Algorithms may slightly exceed `max_fe` upon completing their final generation batch; use `result.evaluations` to record actual function evaluations rather than assuming exact budget alignment.
