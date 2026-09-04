# CPU and NPU Performance Specifications

## Two Classes of Performance Metrics

### Single-Task Latency

The execution time required for a single algorithm or operator run with dedicated access to specified hardware resources. The CPU is fixed at 8 threads, and the NPU is allocated exclusively to one benchmark worker per card, without resource sharing from concurrent processes. Single-task speedup must be computed using this baseline:

```text
speedup = median_cpu8_seconds / median_npu_seconds

```

### System Throughput

The total number of completed independent runs per hour under parallel multi-task execution on a server. CPU throughput can scale across multiple 8-thread processes via `joblib`, while NPU throughput can scale with one or capacity-filtered multiple processes per card. Throughput improvements must not be reported as substitutes for single-task speedup.

## Standard Timing Protocol

1. Fix the algorithm, problem, $N$, $\text{max\_fe}$, random seed, and precision (`dtype`) policy.

2. Execute an initial warm-up run with identical tensor shapes to exclude one-time compilation overhead from steady-state measurements.

3. Clear or record cache states without introducing asymmetric optimization advantages for either device.

4. Synchronize the NPU immediately before starting the timer.

5. Measure only algorithm optimization time, excluding module imports, environment initialization, and artifact compression.

6. Synchronize the NPU immediately after stopping the timer.

7. Log complete population snapshots, metrics, host memory, and device memory.

8. Execute at least 20 independent runs and report the median values.

The standard workflow automatically implements steps 4 and 6:

```python
result = optimize(
    algorithm,
    problem,
    device="npu:0",
    seed=seed,
    synchronize_timing=True,
)

```

## CPU Threading Configuration

CPU benchmarks must explicitly allocate 8 intra-op compute threads and configure corresponding linear algebra libraries:

```bash
export ASCENDMOEA_CPU_THREADS=8
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export NUMEXPR_NUM_THREADS=8

```

Higher thread counts do not necessarily improve performance. For small tensor shapes, thread synchronization overhead and cache contention may cause single-threaded execution to run faster. Conduct thread scaling sweeps across 1, 2, 4, 8, and 16 threads prior to benchmarking; if the CPU8 protocol is selected, fix threads at 8 and document both the thread scaling curve and the rationale for the configuration.

## Preventing NPU Context Initialization by CPU Workers

CPU processes must set the following environment variables prior to PyTorch backend discovery:

```bash
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
export ASCENDMOEA_DEVICE=cpu

```

Verify via `npu-smi` after execution: CPU worker processes must not appear on any NPU card. Inadvertent NPU context allocations introduce CANN initialization overhead into CPU measurements and inflate host memory consumption due to background compilation pools.

## NPU Synchronization and Host-Device Transfers

Operator microbenchmarks must transfer input tensors to the NPU and synchronize before starting the timer. When isolating operator kernel latency, exclude CPU-to-NPU transfer time; for end-to-end algorithm benchmarks, include all runtime transfers executed during optimization. Both benchmark setups must be labeled distinctly:

```python
x_npu = x_cpu.to("npu:0")
synchronize("npu:0")
started = time.perf_counter()
y = operator(x_npu)
synchronize("npu:0")
elapsed = time.perf_counter() - started

```

Calls to `.item()`, `.cpu()`, or `.numpy()` trigger implicit device synchronizations. Scalar extraction should be restricted to generation boundaries to avoid blocking hardware execution queues within generational loops.

## Sizing $N$ and $\text{max\_fe}$

Settings of $N=100$ and $\text{max\_fe}=10,000$ are suitable for baseline quality validation and functional checks, but they are generally insufficient to demonstrate NPU acceleration. Determine performance benchmarking configurations via preliminary scaling sweeps:

| Scale Level | Suggested $N$ | Suggested $\text{max\_fe}$ | Evaluation Focus |
| --- | --- | --- | --- |
| Quality Baseline | 100 | 10,000 | Functional comparison against reference implementations
| Medium Load | 500 | 50,000 | Identifying parallelism crossover thresholds
| High Load | 1,000 | 100,000 | Benchmarking large-batch hardware throughput
| Operator Stress | 2,000 to 10,000 | N/A | Stress-testing $\mathcal{O}(N^2)$ operators and memory limits

Determine benchmark configurations empirically. If NPU speedup over CPU8 is negligible, verify the following in sequence: whether tensors are placed on the NPU, whether implicit host synchronizations occur, whether batch sizes are too small, whether iterative loops remain in objective functions, whether initial compilation overhead was included, whether cards are shared, and whether $\mathcal{O}(N^2)$ memory limits caused degradation. Increase benchmark scale only after resolving implementation bottlenecks.

## Multi-Task Concurrency on NPU

Establish a single-task baseline with 1 worker per card before evaluating 2, 3, or 4 concurrent worker processes. If median single-task runtime degrades beyond a specified tolerance threshold (recommended: 5%) relative to exclusive execution, maintain 1 worker per card for latency benchmarks. For throughput experiments, select the worker concurrency level that maximizes hourly completed tasks while explicitly documenting shared execution modes.

## Numerical Precision Guidelines

CPUs default to `float64`, whereas NPUs default to `float32` because certain Ascend operators lack native `float64` support. Numerical consistency across hardware platforms must be validated as follows:

* Validate arithmetic outputs using combined tolerance checks: $\text{atol} + \text{rtol} \times \vert{}\text{reference}\vert{}$.

* Evaluate non-dominated sorting and selection independently via front indices, rank memberships, and deterministic tie-breaking logic.

* Compare statistical distributions of quality metrics across 20 runs rather than expecting identical per-generation trajectories from identical seeds.

* Perform sensitivity analyses on tolerance boundaries for solutions located near active constraint surfaces.

## Required Reporting Metrics

* Raw wall-clock runtimes for all individual CPU8 and NPU runs

* Warm-up duration and an explicit statement on whether it was included in timing

* Single-task speedup ratios and system throughput ratios

* Peak host PSS and RSS for CPU and NPU configurations

* Peak device HBM consumption on NPU

* Generational logs of evaluations (`FE`), population snapshots, and quality indicators

* Failure logs, timeouts, restart records, and exception traces

* Time-series profiles of hardware utilization
