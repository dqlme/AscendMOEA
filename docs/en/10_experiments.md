# Metrics, Experiments, and Monitoring

## Quality Indicators API

```python
from ascendmoea.metrics import (
    coverage_rate,
    cpf,
    delta_p,
    dm,
    feasible_rate,
    gd,
    hv,
    igd,
    igdp,
    igdx,
    psp,
)

```

| Metric | Optimization Goal | Required Reference Data | Evaluation Focus |
| --- | --- | --- | --- |
| HV | Maximization | Objective-space reference front / normalization bounds | Combined convergence and distribution spread
| IGD | Minimization | Objective-space reference points | Coverage across the Pareto front
| IGD+ | Minimization | Objective-space reference points | Dominance-compliant distance metric
| GD | Minimization | Objective-space reference points | Distance from solutions to true Pareto front
| DeltaP | Minimization | Objective-space reference points | Bidirectional distance balance
| DM | Maximization | Objective-space reference points | Grid-based distribution diversity
| CPF | Maximization | Objective-space reference points | Coverage along the Pareto front manifold
| IGDX | Minimization | Decision-space reference Pareto set | Quality of multimodal decision-space solutions
| CR | Maximization | Decision-space reference Pareto set | Decision space coverage ratio
| PSP | Minimization | Decision-space reference Pareto set | Comprehensive decision-space distance and spread
| Feasible rate | Maximization | None | Proportion of feasible solutions under constraints
Hypervolume calculations for problems with more than three objectives ($M > 3$) rely on Monte Carlo estimation. Formal comparative studies must report random seeds, sample point counts, and normalization bounds; if exact high-dimensional HV metrics are required, link an external deterministic solver and record its release version.

## Generational Indicator Tracking

```python
from ascendmoea import HistoryMonitor, optimize
from ascendmoea.metrics import hv, igd

history = HistoryMonitor(copy_to_cpu=True)
result = optimize(algorithm, problem, monitors=history, device="cpu", seed=1)
reference = problem.optimum.detach().cpu()

rows = []
for record in history.records:
    rows.append(
        {
            "fe": record.evaluations,
            "hv": hv(record.population, reference),
            "igd": igd(record.population, reference),
        }
    )

```

`HistoryMonitor` retains all generational populations in memory, which is suitable only for interactive debugging or small-scale problems. Production benchmark experiments must implement streaming monitors: serialize `decs`, `objs`, `cons`, evaluation counts, and timestamps directly to disk per generation using temporary files and atomic rename operations, updating execution progress in an index file.

## Recommended Results Directory Structure

```text
results/
  <stage>/<device>/<algorithm>/<problem>/seed_<seed>/
    config.json
    environment.json
    progress.json
    generations/
      fe_00000100.npz
      fe_00000200.npz
    metrics.csv
    summary.json
    resource_samples.csv
    COMPLETE

```

The `COMPLETE` marker file must be created only after all arrays are flushed to disk, the summary JSON validates successfully, and file integrity checks pass. Retain error logs for failed executions; re-runs must write to distinct execution directories to preserve traceable experiment histories.

## Minimal Schema for `summary.json`

```json
{
  "schema_version": 1,
  "algorithm": "NSGAII",
  "problem": "DTLZ2",
  "device": "npu:0",
  "population_size": 100,
  "max_fe": 10000,
  "seed": 1001,
  "elapsed_seconds": 12.345,
  "peak_host_rss_bytes": 123456789,
  "peak_host_pss_bytes": 120000000,
  "peak_device_memory_bytes": 987654321,
  "metrics": {"HV": 0.7, "IGD": 0.02},
  "status": "complete"
}

```

Record accompanying metadata including git commit hash, package version, CANN/PyTorch/`torch_npu` versions, CPU/NPU hardware specifications, thread counts, worker concurrency limits, warm-up iterations, and device synchronization modes.

## Memory Profiling

### Host System Memory

Sample process-tree Resident Set Size (RSS) and Proportional Set Size (PSS) at intervals between 0.2 and 1.0 seconds:

* RSS is readily accessible, but summing across multiple worker processes counts shared libraries multiple times.

* PSS scales shared memory proportionally, offering a more accurate representation of total memory footprint across multiple concurrent workers.

* Track system-wide available memory, swap utilization, and out-of-memory (OOM) kernel events simultaneously.

On Linux platforms, parse PSS from `/proc/<pid>/smaps_rollup`. Worker metrics must aggregate child processes to avoid omitting sub-processes spawned during model compilation or background data loading.

### NPU Device Memory

Poll `npu-smi info` periodically to record High Bandwidth Memory (HBM) usage by device ID and PID. Report:

* Peak HBM per individual worker process

* Aggregated peak HBM across all concurrent worker processes on a given NPU card

* Global device peak allocation throughout the experimental stage

* Incidents of accidental NPU context initialization by CPU worker processes

Memory samplers must run as asynchronous background processes and maintain identical sampling frequencies across CPU and NPU runs. Benchmark background sampler overhead first to ensure high-frequency polling does not skew short-running operator timings.

## Statistical Protocols for Replicated Experiments

Execute at least 20 independent random seed runs per algorithm/problem/device/software configuration. Persist raw run matrices before computing median, mean, standard deviation, interquartile range (IQR), and 95% confidence intervals. Evaluate solution quality using paired non-parametric statistical tests with multiple-comparison corrections; report execution runtime using both per-run latency and throughput per function evaluation (FE).

Do not rely solely on aggregate mean values; raw run logs are necessary to diagnose outlier runs, feasible ratio degradation, and long-tail execution latencies.

## Visualization Standards

`ascendmoea.plotting` generates Plotly figure objects for objective spaces, decision spaces, and metric convergence curves. Publication-quality figures should adhere to the following standards:

* Save one plot per file; do not merge distinct runs into multi-panel layouts within a single graphic output.

* Use Times New Roman typography across all figure labels and annotations.

* Export raster images (PNG) at a minimum resolution of 300 PPI.

* Export vector graphics (PDF) using vector-capable rendering backends, confirming that internal graphics primitives are not converted to full-page raster bitmaps.

* Maintain generating scripts, raw data hashes, and figure caption metadata for every generated plot.

```python
from ascendmoea.plotting import draw_metric_curve

figure = draw_metric_curve(fes, values, "IGD")
figure.write_html("igd_curve.html")
figure.write_image("igd_curve.pdf")
figure.write_image("igd_curve.png", scale=4)

```
