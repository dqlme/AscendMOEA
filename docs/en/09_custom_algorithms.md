# Custom Algorithm Development

## Minimal Algorithm Architecture

Custom algorithms inherit from `Algorithm`, managing the outer generational lifecycle while delegating intra-generation processing to vectorized tensor operators:

```python
import torch

from ascendmoea import Algorithm, register_algorithm
from ascendmoea.operators import crowding_distance, merge_pop, nd_sort, operator_ga

def environmental_selection(population, size):
    front_no, last_front = nd_sort(
        population.objs, population.cons, n_sort=size
    )
    crowding = crowding_distance(population.objs, front_no)
    selected = front_no < last_front
    remaining = size - int(selected.sum().item())
    if remaining:
        candidates = torch.where(front_no == last_front)[0]
        order = torch.argsort(crowding[candidates], descending=True, stable=True)
        selected[candidates[order[:remaining]]] = True
    return population[selected]

@register_algorithm(name="BatchGA")
class BatchGA(Algorithm):
    """Simple batch-oriented multi-objective genetic algorithm."""

    name = "BatchGA"

    def _solve(self, problem, output, should_stop, should_pause):
        population = problem.initialization()
        while self.not_terminated(
            problem, population, output, should_stop, should_pause
        ):
            offspring = problem.evaluate(operator_ga(problem, population.decs))
            population = environmental_selection(
                merge_pop(population, offspring), problem.N
            )

```

## Termination and Generational Protocol

The `not_terminated` method must be evaluated at every generation boundary on the active population. This method is responsible for:

* Updating `algorithm.result` and `final_population`

* Invoking workflow callbacks and attached runtime monitors

* Evaluating external interrupt triggers, pause states, and evaluation ceilings (`problem.max_fe`)

* Ensuring final optimization results can be extracted via the public API

Do not substitute this check with raw `problem.FE` loops, as bypassing `not_terminated` disables population history tracking and monitor events.

## Vectorization Boundaries

A single outer generational `while` loop is standard design. Intra-generation operations must be fully vectorized:

* Mating selection and indexing

* Dominance evaluation, non-dominated sorting, and fitness assignment

* Crowding distance calculation, neighborhood distance, and reference line association

* Variation operators, mutation, and boundary clamping

* Subproblem weight vector updates and environmental selection

* Batched objective and constraint evaluations

Sequential truncation schemes that strictly depend on state modifications from preceding steps may retain minimal inner loops, provided each underlying distance step is vectorized and CPU-accelerated paths are exposed where applicable. Implementations must preserve stable sorting and deterministic pseudo-random number generator (PRNG) usage.

## Device-Agnostic Implementation

Anti-pattern:

```python
mask = torch.zeros(population.objs.shape[0], dtype=torch.bool)

```

Correct implementation:

```python
mask = torch.zeros(
    population.objs.shape[0], dtype=torch.bool, device=population.objs.device
)

```

Prefer deriving tensor allocations directly from existing structures, e.g., `torch.zeros_like(front_no, dtype=torch.bool)`. Avoid importing `torch_npu` directly within algorithm files; device backend initialization is managed by the workflow runtime.

## State Management and Caching

Constructor parameters must be bound to instance attributes. Invoking `solve` clears the base `result` structure and `metric_cache`; reset any algorithm-specific runtime caches at the beginning of `_solve`. Do not store mutable tensor buffers in class-level attributes.

When maintaining reference vectors, neighborhood topologies, or surrogate models:

* Initialize buffers only after the target problem device is configured.

* Inherit precision types directly from `problem.dtype`.

* Avoid reusing dimension-specific structural tensors across different problem instances.

* Track model training time and memory footprint separately to explain algorithmic overhead.

## Component Registration and Extension Packages

Use the decorator API for project-level extensions:

```python
from ascendmoea import create_algorithm

algorithm = create_algorithm("BatchGA", save=10)

```

External third-party packages should register components upon package import using distinct, non-conflicting naming keys. The component registry rejects duplicate identifiers by default; pass `replace=True` only during intentional compatibility migrations.
