# Built-in Problem API

## Instantiation and Registry Queries

Problems can be imported directly or created via registered identifiers:

```python
from ascendmoea import create_problem, get_problem, list_problems
from ascendmoea.problems import DTLZ2

direct = DTLZ2(n=100, m=3, d=12, max_fe=10_000)
configured = create_problem("DTLZ2", n=100, m=3, d=12, max_fe=10_000)
assert get_problem("DTLZ2") is DTLZ2
print(list_problems())

```

Registry lookups are case-sensitive. Identifiers that represent valid Python symbols are exported directly by name in `ascendmoea.problems`; configuration pipelines should store registered string identifiers.

## Problem Suite Constructor Signatures

| Problem Suite | Public Parameters | Description |
| --- | --- | --- |
| `ZDT1` to `ZDT6` | `n=100, d=None, max_fe=10000` | Fixed bi-objective problems; `d=None` resolves to suite defaults.
| `DTLZ1` to `DTLZ7` | `n=100, m=3, d=None, max_fe=10000` | `d=None` derives dimensions from problem index and $M$.
| `CF1` to `CF10` | `n=100, m=2, d=10, max_fe=10000` | Objective counts are defined by specific problem formulations.
| `DASCMOP1` to `DASCMOP9` | `n=100, d=30, max_fe=10000` | Suites 1–6 are bi-objective; suites 7–9 are tri-objective.
| `LIRCMOP1` to `LIRCMOP14` | `n=100, d=30, max_fe=10000` | Suites 1–12 are bi-objective; suites 13–14 are tri-objective.
| `SMOP1` to `SMOP8` | `n=100, m=2, d=100, max_fe=10000, theta=0.1` | `theta` configures Pareto-optimal sparsity level.
| `LSMOP1` to `LSMOP9` | `n=100, m=3, d=None, max_fe=10000, nk=5` | Defaults to $D = 100 \times M$; `nk` sets sub-component grouping count.
| Multimodal Suites | `n=100, max_fe=10000` | Objectives, dimensions, bounds, and reference sets are fixed by problem definition.
| `RWMOP1` to `RWMOP10` | `n=100, max_fe=10000` | Dimensions and objectives are fixed by engineering problem specifications.
| `Sparse_PM/PO/SR` | `n=100, max_fe=10000` | Dataset dimensions and matrices are fixed.

Combinatorial benchmark specifications:

| Registered Name | Constructor Signature | Encodings and Constraints |
| --- | --- | --- |
| `Sparse_KP` | `(n=100, m=2, d=250, max_fe=10000)` | Binary; accepts only $M=2, D=250$.
| `MOTSP` | `(n=100, m=2, d=30, max_fe=10000)` | Permutation; accepts only $M=2, D=30$.
| `mQAP` | `(n=100, m=2, d=10, max_fe=10000)` | Permutation; accepts only $M=2, D=10$.
| `MONRP` | `(n=100, d=100, max_fe=10000)` | Binary; accepts only $D=100$.
| `MPDMP` | `(n=100, m=10, max_fe=10000, lower=-100, upper=100)` | Continuous 2D decisions; requires $M \ge 3$.

Passing unsupported `m` or `d` values to data-fixed problem suites raises a `ValueError`; datasets are not rescaled automatically.

## Complete Registry of 155 Problem Identifiers

### General Benchmark Suites (13)

```text
ZDT1, ZDT2, ZDT3, ZDT4, ZDT5, ZDT6,
DTLZ1, DTLZ2, DTLZ3, DTLZ4, DTLZ5, DTLZ6, DTLZ7

```

### Constrained Benchmark Suites (33)

```text
CF1, CF2, CF3, CF4, CF5, CF6, CF7, CF8, CF9, CF10,
DASCMOP1, DASCMOP2, DASCMOP3, DASCMOP4, DASCMOP5, DASCMOP6,
DASCMOP7, DASCMOP8, DASCMOP9,
LIRCMOP1, LIRCMOP2, LIRCMOP3, LIRCMOP4, LIRCMOP5, LIRCMOP6,
LIRCMOP7, LIRCMOP8, LIRCMOP9, LIRCMOP10, LIRCMOP11, LIRCMOP12,
LIRCMOP13, LIRCMOP14

```

### Sparse and Large-Scale Suites (17)

```text
SMOP1, SMOP2, SMOP3, SMOP4, SMOP5, SMOP6, SMOP7, SMOP8,
LSMOP1, LSMOP2, LSMOP3, LSMOP4, LSMOP5, LSMOP6, LSMOP7, LSMOP8, LSMOP9

```

### Multimodal and Real-World Benchmark Suites (74)

```text
MMF1, MMF1_e, MMF1_z, MMF2, MMF3, MMF4, MMF5, MMF6, MMF7, MMF8,
MMF9, MMF10, MMF11, MMF12, MMF13, MMF14, MMF14_a, MMF15, MMF15_a,
SYM_PART_simple, SYM_PART_rotated, Omni_test,
IDMPM2T1, IDMPM2T1_e, IDMPM2T2, IDMPM2T2_e, IDMPM2T3, IDMPM2T3_e,
IDMPM2T4, IDMPM2T4_e,
IDMPM3T1, IDMPM3T1_e, IDMPM3T2, IDMPM3T2_e, IDMPM3T3, IDMPM3T3_e,
IDMPM3T4, IDMPM3T4_e,
IDMPM4T1, IDMPM4T2, IDMPM4T3, IDMPM4T4,
MMMOP1A, MMMOP1B, MMMOP2A, MMMOP2B,
MMMOP3A, MMMOP3B, MMMOP3C, MMMOP3D,
MMMOP4A, MMMOP4B, MMMOP4C, MMMOP4D,
MMMOP5A, MMMOP5B, MMMOP5C, MMMOP5D,
MMMOP6A, MMMOP6B, MMMOP6C, MMMOP6D,
poly_2, poly_3, poly_4, poly_5, poly_6, poly_8, poly_10, poly_12,
map, application, RLP, UECLP

```

### Real-World and Combinatorial Applications (18)

```text
RWMOP1, RWMOP2, RWMOP3, RWMOP4, RWMOP5,
RWMOP6, RWMOP7, RWMOP8, RWMOP9, RWMOP10,
Sparse_KP, Sparse_PM, Sparse_PO, Sparse_SR,
MOTSP, mQAP, MONRP, MPDMP

```

## Encoding and Evaluation Contract

The entries in `problem.encoding` map directly to decision variable columns: 1 for Real, 2 for Integer, 3 for Categorical Label, 4 for Binary, and 5 for Permutation. `initialization()` generates complete batches conforming to these encodings. `evaluate()` applies boundary clamping, repairs discrete variables, and returns a `Population` instance. Feasibility is evaluated as $g(x) \le 0$.

```python
population = problem.initialization()
reevaluated = problem.evaluate(population.decs[:8])
assert reevaluated.decs.shape == (8, problem.D)
assert reevaluated.objs.shape == (8, problem.M)

```

The problem instance accumulates evaluated counts in `FE`. A problem instance must not be shared across concurrent threads or reused across independent experimental runs.

## Reference Data Specifications

| Property | Scope | Purpose |
| --- | --- | --- |
| `problem.optimum` | All problems | Reference data for objective-space quality indicators
| `problem.pf` | Problems with explicit geometry | Drawable Pareto front representation; may be `None`<br> |
| `problem.ps` | Built-in multimodal problems | Reference Pareto sets for IGDX, CR, and PSP metrics
| `problem.reference_point` | Built-in multimodal problems | Standard reference point defined in literature; may be `None`<br> |

Reference data structures automatically migrate alongside the problem to the assigned target device. Datasets for real-world and multimodal suites are resolved from packaged MAT assets; install the library via standard wheel distributions rather than executing raw, unbundled scripts.
