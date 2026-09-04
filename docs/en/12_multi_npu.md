# Multi-NPU Batch Scheduling

## Scheduling Architecture

AscendMOEA's `Workflow` orchestrates single-run executions. Multi-process parallel scheduling is managed externally, isolating random states, evaluation counters, and device contexts across processes while simplifying retry logic and artifact segregation.

Concurrency strategies differ across hardware:

* CPU: Fix 8 threads per process and parallelize across independent random seeds using `joblib`.

* NPU: Launch one (or a capacity-validated small number of) independent worker processes per physical accelerator.

* Do not share algorithm, problem, or NPU device instances across threads or worker processes.

## Single-Node 8-NPU Execution Example

`run_one.py` reads target device parameters from the environment:

```python
import os

from ascendmoea import create_algorithm, create_problem, optimize

card = int(os.environ["ASCENDMOEA_NPU_INDEX"])
algorithm = create_algorithm(os.environ["ALGORITHM"], save=1)
problem = create_problem(
    os.environ["PROBLEM"],
    n=int(os.environ["POPULATION_SIZE"]),
    max_fe=int(os.environ["MAX_FE"]),
)
result = optimize(
    algorithm,
    problem,
    device=f"npu:{card}",
    seed=int(os.environ["SEED"]),
)

```

Launch isolated worker processes across distinct physical cards:

```bash
for card in 0 1 2 3 4 5 6 7; do
  ASCENDMOEA_NPU_INDEX=$card \
  ASCEND_RT_VISIBLE_DEVICES=$card \
  ALGORITHM=NSGAII PROBLEM=DTLZ2 \
  POPULATION_SIZE=1000 MAX_FE=100000 SEED=$((1001 + card)) \
  python run_one.py >"worker_${card}.log" 2>&1 &
done
wait

```

Because environment variable semantics for visible devices may vary across CANN releases, verify behavior against the installed environment. Explicitly pass `npu:k` in application code and log mappings between logical identifiers and physical cards.

## CPU Parallel Execution with joblib

```python
from joblib import Parallel, delayed

def run_seed(seed):
    return optimize(
        create_algorithm("NSGAII", save=1),
        create_problem("DTLZ2", n=100, max_fe=10_000),
        device="cpu",
        cpu_threads=8,
        seed=seed,
    ).elapsed_seconds

times = Parallel(n_jobs=16, backend="loky")(
    delayed(run_seed)(seed) for seed in range(1001, 1021)
)

```

The aggregate concurrency `n_jobs * cpu_threads` must not exceed validated physical limits verified via NUMA profiling. Pin workers to dedicated CPU core sets to prevent cross-NUMA memory transfers and scheduler drift.

## Reliable Background Execution

Long-running batch queues should be managed via systemd, Slurm, or equivalent workload managers rather than interactive SSH sessions. A production systemd unit should:

* Define a static working directory and environment file.

* Configure `Restart=on-failure` with restart rate limits.

* Route stdout and stderr to persistent log files.

* Run an independent, read-only monitoring timer.

* Trigger alerts when progress files remain unupdated beyond a timeout threshold, without terminating active processes automatically.

* Skip previously finished runs by checking for valid `COMPLETE` marker files.

## Task Manifest Files

Generate an immutable JSON Lines manifest prior to queue execution, specifying one run per line:

```json
{"algorithm":"NSGAII","problem":"DTLZ2","seed":1001,"n":1000,"max_fe":100000,"device":"npu:0"}

```

Workers update execution progress via atomic progress files. Upon restarting a batch queue, verify existing summary files and checksums before skipping tasks; do not infer task completion based solely on directory existence.

## Monitoring Frequency

For stable long-running queues, set external monitoring checks to a maximum interval of 30 minutes. Recommended best practices: execute lightweight, read-only local health checks every 5 minutes, and aggregate metrics on remote controllers every 20 to 30 minutes:

* Service operational status and queue heartbeats

* Completed worker count, error counts, and oldest active timestamp

* Per-NPU active processes, utilization, operating temperature, and HBM consumption

* CPU load averages, PSS memory, available host memory, and swap utilization

* Disk space and readable result files

* Runtime latency outliers and non-finite (NaN/Inf) metric detections

If a software bug is detected, pause and re-queue only the affected experimental stage while allowing independent production queues to continue execution.
