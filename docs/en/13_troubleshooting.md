# Troubleshooting Installation and Execution

This guide covers common issues encountered during AscendMOEA installation, device configuration, performance benchmarking, and batch experimentation. Before debugging, preserve the configuration files, complete execution logs, and input datasets of failed runs, and reproduce the failure in an isolated directory to avoid overwriting existing experimental artifacts.

## Recommended Troubleshooting Sequence

Diagnose issues across the following four layers in order:

1. **Software Environment**: Verify Python environment integrity, dependency package versions, and module search paths.

2. **Compute Devices**: Confirm CPU/NPU hardware availability and ensure execution runs on the assigned target device.

3. **Numerical Computation**: Validate tensor dtypes, tensor shapes, constraint definitions, and inspect for anomalous values (NaN/Inf).

4. **Algorithmic Execution**: Verify evaluation counters, termination criteria, population updates, and attached monitoring hooks.

Log diagnostic commands and their terminal outputs at each step to maintain a traceable debugging history.

## Installation and Import Issues

### Cannot Import `ascendmoea`

A `ModuleNotFoundError: No module named 'ascendmoea'` error typically indicates that AscendMOEA is not installed in the active environment or that the Python executable does not match the installation target.

Run the following checks:

```bash
which python
python -m pip show AscendMOEA
python -c "import ascendmoea; print(ascendmoea.__file__)"

```

If `pip show` returns no output, reinstall the package from the repository root:

```bash
python -m pip install .

```

When using virtual environments, ensure the target environment is activated before installing or executing scripts.

### Missing MAT Data Files

Certain real-world problems require MATLAB `.mat` data files bundled with the package distribution. If problems execute within the raw source tree but fail when installed as a package due to missing data files, check the following:

* Ensure `ascendmoea/vendor/**/*.mat` assets are included in the build distribution.

* Confirm package data inclusions are properly declared in `pyproject.toml`.

* Verify that data loaders use Python package resource APIs (`importlib.resources`) rather than relative working-directory paths.

* Verify that the active environment is running the latest build artifact.

Rebuild and reinstall the package after modifying package asset configurations.

## NPU Hardware Unavailable

Check hardware status and framework bindings:

```bash
npu-smi info
python -c "import torch, torch_npu; print(torch.npu.is_available())"

```

If `npu-smi info` fails to detect accelerators, verify kernel drivers and device firmware. If the device appears in `npu-smi` but `torch.npu.is_available()` evaluates to `False`, check the following in order:

1. Verify CANN installation integrity.

2. Confirm CANN environment variable scripts have been sourced.

3. Validate version compatibility between PyTorch and `torch_npu`.

4. Ensure the Python interpreter version is supported by the installed software stack.

5. Verify that the current user account has read/write permissions for NPU device nodes.

Do not overwrite the NPU-specific PyTorch distribution with generic CPU-only or CUDA-enabled wheels. Consult CANN and `torch_npu` compatibility matrices before modifying runtime packages.

## CPU Tasks Allocating Memory on NPU 0

If CPU worker processes appear in the `npu-smi` process table, PyTorch is automatically initializing device backends. Set the following environment variables prior to importing PyTorch:

```bash
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
export ASCENDMOEA_DEVICE=cpu

```

Ensure the optimization workflow explicitly selects the CPU target:

```python
result = optimize(algorithm, problem, device="cpu", seed=42)

```

Restart the Python process after modifying environment variables. Benchmark runs that included NPU backend initialization overhead during CPU testing must be rerun to preserve valid latency and memory statistics.

## NPU Performance Lower than 8-Thread CPU

NPU execution does not guarantee speedups across all problem configurations. Small population sizes, low evaluation budgets, or frequent host-device synchronizations can cause dispatch overhead to exceed parallel tensor speedups.

Diagnose performance using the following checklist:

1. Confirm that algorithms, problems, and core tensors reside on the target NPU.

2. Ensure CPU and NPU configurations use identical population sizes and evaluation ceilings.

3. Run a warm-up step to exclude one-time operator compilation from timing measurements.

4. Synchronize the NPU immediately before starting and after stopping timers.

5. Check for frequent calls to `.item()`, `.cpu()`, or `.numpy()` inside inner optimization loops.

6. Confirm that objective functions and evolutionary operators evaluate entire population batches vectorially.

7. Avoid allocating CPU tensors inside iterative loops followed by runtime transfers to NPU.

8. Verify that no background processes are contending for compute cores or HBM on the same NPU.

9. Inspect device compute utilization, memory usage, and power draw using `npu-smi`.

10. Identify primary execution bottlenecks via profiling tools or sectional timers.

Apply targeted optimizations—such as batching operations, minimizing device synchronizations, preallocating immutable buffers, or scaling benchmark problem sizes—once bottlenecks are located. Increasing the evaluation ceiling (`max_fe`) alone will not resolve computational inefficiencies in per-iteration routines.

## Memory Consumption Anomalies

### Aggregated Multi-Process RSS Exceeds Physical RAM

RSS counts shared libraries and memory-mapped pages redundantly across multiple processes; summing RSS across workers produces an inflated estimate of physical memory consumption. Benchmark pipelines should track Proportional Set Size (PSS) and categorize memory by process type:

* Main coordinator process

* Parallel worker processes

* CANN compilation or background runtime daemons

* Logging and monitoring processes

If memory growth persists, log the PSS, RSS, and process count for each category to locate the source of the leak.

### Continuous Growth in NPU HBM Allocation

Check for the following common root causes:

* Monitors retaining un-detached NPU tensors across generations.

* Algorithm-level caches appending generational data without releasing obsolete buffers.

* Python data structures or callback closures retaining references to out-of-scope tensors.

* Redundant allocations of large intermediate tensors within iterative loops.

Population records intended for persistent storage should be detached, copied to host memory (`.cpu()`), and written incrementally to disk. Never accumulate full generational histories directly within NPU memory.

### Out-of-Memory Errors from $\mathcal{O}(N^2)$ Operators

Operators such as dominance matrix construction and pairwise distance calculations scale with spatial complexity $\mathcal{O}(N^2)$. Memory requirements scale quadratically as population size increases.

When debugging out-of-memory errors, record:

* Population sizes and tensor shapes

* Precision data types (`dtype`)

* Intermediate and input tensor allocation counts

* Peak device memory allocation

* The specific operator executing at the point of failure

Mitigate memory limits by eliminating redundant intermediate tensors, chunking matrix computations into block partitions, or reducing precision where appropriate. If a benchmark configuration exceeds device memory limits, document the failure conditions explicitly; do not modify population sizes, selection rules, or truncation parameters without documenting the change.

## Metric Calculation Anomalies

| Observed Behavior | Diagnostic Checks |
| --- | --- |
| Metric evaluates to NaN or Inf | Verify objective values, reference Pareto data, normalization denominators, and ensure feasible population subsets are non-empty.
| Hypervolume (HV) is 0 | Inspect reference point bounds, optimization minimization directions, and verify normalized solutions do not fall outside the reference bounding box.
| IGD is abnormally large despite correct front plots | Check objective column ordering, objective counts ($M$), and dimensionality alignment against the reference front.
| IGDX exhibits anomalous values | Confirm that input reference data corresponds to decision-space Pareto sets rather than objective-space Pareto fronts.
| Feasible rate remains 0 throughout execution | Ensure constraint violations follow the $g(x) \le 0$ formulation and confirm tolerance thresholds for equality constraints are appropriately calibrated.

When comparing metrics across compute devices, verify that CPU and NPU runs use identical floating-point precision and tolerance settings.

## Unresponsive Batch Tasks

Extended task duration does not necessarily mean the process has hung. Validate system activity across multiple indicators:

* Last modified timestamp of active progress files

* Outgoing write activity in log files

* Process CPU utilization and I/O throughput

* NPU core utilization and device memory allocations

* Historical runtimes for matching algorithm-problem configurations

* Active operator compilation or high-overhead metric evaluations

Declare a task unresponsive only when execution time exceeds configured timeouts and progress files, logs, CPU, I/O, and NPU metrics show no activity. Before terminating tasks, archive the process tree state, stdout/stderr logs, hardware status, and run configurations.

## Minimal Reproducible Example

Before submitting an issue report, isolate the bug into a minimal reproducible example. Include the following artifacts:

* A single algorithm and a single problem definition

* A fixed random seed

* Explicit device specification (`cpu` or `npu:k`)

* The minimal population size and evaluation count required to trigger the failure

* Version details for AscendMOEA, Python, PyTorch, `torch_npu`, and CANN

* Complete stack traces, error logs, and the exact terminal execution command

Following a bug fix, re-run only the affected experimental runs while preserving original failure artifacts for validation.
