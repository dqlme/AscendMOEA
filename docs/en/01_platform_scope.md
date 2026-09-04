# Platform Scope and Boundaries

## Target Problems

AscendMOEA targets evolutionary optimization tasks involving multiple conflicting objectives. It represents decision variables, objective values, and constraint violations across a population as batched tensors. Key generational operations—including sorting, selection, crossover, mutation, distance computations, and problem evaluations—are converted into vectorized operations wherever possible. The identical algorithm implementation runs seamlessly on both CPUs and Huawei Ascend NPUs.

The platform primarily supports four research domains:

* Unconstrained multi-objective optimization
* Constrained multi-objective optimization
* Multimodal multi-objective optimization
* Sparse large-scale multi-objective optimization

It also supports combinatorial and real-world engineering problems, extending beyond standard analytical benchmark suites to data-driven tasks.

## Design Principles

### Batched Evaluation First

Problem methods `cal_obj` and `cal_con` accept batched decision tensors of shape `[N, D]` and must return an `[N, M]` objective tensor and an `[N, C]` constraint tensor, respectively. Avoid invoking Python functions per individual solution, as this breaks NPU parallelism and amplifies interpreter and dispatch overheads.

### Decoupling Algorithms from Problems

Algorithms depend strictly on the `Problem` protocol rather than concrete problem implementations. The problem encapsulates boundaries, encodings, initialization, evaluations, and reference data; the algorithm governs offspring generation, environmental selection, and termination criteria.

### Composable Public Operators

Operators in `ascendmoea.operators` serve built-in algorithms and can be directly reused in custom user algorithms. The public API eliminates dependencies on internal source paths.

### Explicit Timing Synchronization

Ascend operator dispatches are asynchronous. By default, `Workflow` performs explicit device synchronizations immediately before and after execution timing, ensuring that reported wall-clock times reflect actual hardware computations rather than Python submission latency.

## Architecture Overview

```mermaid
flowchart LR
    U["User / Experiment Scheduler"] --> W["Workflow"]
    W --> A["Algorithm"]
    W --> P["Problem"]
    A --> O["Tensorized operators"]
    A --> POP["Population"]
    P --> POP
    W --> M["Monitor"]
    POP --> Q["Metrics and result storage"]
    D["CPU / Ascend NPU"] --> A
    D --> P
    D --> O

```

## Current Limitations

* Built-in algorithms focus on multi-objective evolutionary algorithms (MOEAs); general-purpose gradient-based optimizers are not provided.
* `Workflow` orchestrates a single algorithm execution run; multi-process and multi-NPU batch experiments must be managed by external schedulers.
* NPU execution requires pre-installed CANN, PyTorch, and `torch_npu` runtimes matching the target environment; these runtime packages are not bundled with the distribution.
* Quality indicators require valid reference Pareto fronts or Pareto sets; metrics computed without reference data are not comparable.
* Real-world problem data files are packaged as distribution resources; code execution must not assume that the current working directory is the source root.
* High-dimensional Hypervolume (HV) relies on Monte Carlo estimation; comparative studies must fix random seeds and report sampling parameters.

## When to Use NPU Acceleration

NPU acceleration benefits large-scale, regular, and compute-intensive tensor operations. If population sizes are small, evaluation budgets are limited, Python control flow dominates, or frequent host-device data transfers occur, the CPU may be faster. Do not assume arbitrary combinations of `N` and `max_fe` will yield speedups; validate scale efficiency via preliminary parameter sweeps before fixing benchmark configurations.

## Reproducibility Requirements

A reproducible experiment must log at least: library version, source commit hash, algorithm and problem parameters, population size, maximum evaluation count, random seeds, device model, software stack versions, CPU thread count, NPU device index, warm-up strategy, synchronization method, wall-clock runtime, peak host memory, peak device memory, final metric values, and complete generational population snapshots. Refer to Chapters 10–12 for detailed experimentation protocols.
