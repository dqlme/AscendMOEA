# Problem Suites and Data Resources

## Problem System Architecture

The release version registers 155 test problems, spanning analytical benchmarks, constrained optimization, multimodal optimization, sparse large-scale optimization, and real-world application suites. The exact catalog of registered problem names should be inspected dynamically via the installed package:

```python
from ascendmoea import list_problems

for name in list_problems():
    print(name)

```

## Major Benchmark Suites

| Category | Problem Suite | Count / Scope | Key Characteristics |
| --- | --- | --- | --- |
| General | ZDT | ZDT1 to ZDT6 | Bi-objective Pareto fronts; non-convex and disconnected geometries
| General | DTLZ | DTLZ1 to DTLZ7 | Scalable objective dimensions and degenerate Pareto fronts
| Large-Scale | LSMOP | LSMOP1 to LSMOP9 | Decision variable grouping and high-dimensional search spaces
| Sparse | SMOP | SMOP1 to SMOP8 | Sparse Pareto-optimal decision vectors
| Constrained | CF | CF1 to CF10 | Nonlinear constraints and fragmented feasible regions
| Constrained | DASCMOP | DASCMOP1 to DASCMOP9 | Adjustable constraint difficulty profiles
| Constrained | LIRCMOP | LIRCMOP1 to LIRCMOP14 | Large infeasible regions and complex constraint landscapes
| Multimodal | MMF | Multiple MMF variants | Multiple Pareto sets and nonlinear mapping topographies
| Multimodal | MMMOP | MMMOP1A to MMMOP6D | Multi-layered decision space landscapes
| Multimodal | IDMP | Base and extended variants | Decision-space equivalent optimal solutions
| Multimodal | SYM-PART, Omni, poly | Multiple variants | Symmetrical, periodic, and polynomial structural mappings
## Real-World and Combinatorial Problems

Real-world problems include the RWMOP1 to RWMOP10 suites, alongside MONRP, MOTSP, MPDMP, RLP, UECLP, application benchmarks, mQAP, and map routing tasks. Sparse engineering suites include `Sparse_KP`, `Sparse_PM`, `Sparse_PO`, and `Sparse_SR`.

These problems typically feature the following operational characteristics:

* Constructors load external MATLAB `.mat` data matrices packaged as package resources.

* Decision variables may follow integer, categorical label, binary, or permutation encodings.

* Configurable decision dimensions ($D$) and objective dimensions ($M$) are restricted by specific data instance files.

* A single evaluation call may require evaluating distance matrices, graph topologies, or combinatorial constraint sets.

Inspect class docstrings and constructor signatures prior to initialization; do not directly apply variable dimension sizing rules ($D$) from analytical functions to data-driven problems.

## Direct Imports and Registry Instantiation

Classes with valid Python identifiers are exported directly through `ascendmoea.problems`:

```python
from ascendmoea.problems import DTLZ2, LIRCMOP5, RWMOP3

p1 = DTLZ2(n=100, m=3, d=12, max_fe=10_000)
p2 = LIRCMOP5(n=100, max_fe=10_000)
p3 = RWMOP3(n=100, max_fe=10_000)

```

All registered problem identifiers can be constructed via the factory API:

```python
from ascendmoea import create_problem

problem = create_problem("SYM_PART_simple", n=100, max_fe=10_000)

```

Identifiers containing spaces, hyphens, or non-standard characters must be instantiated using `create_problem`.

## Variable Encoding Conventions

Each decision variable in `Problem.encoding` is assigned an integer code:

| Value | Encoding Type |
| --- | --- |
| 1 | Real / Continuous
| 2 | Integer
| 3 | Categorical Label
| 4 | Binary
| 5 | Permutation
Base initializers batch-generate candidate decision variables according to `encoding`, applying bound repair and rounding schemes aligned with reference problem definitions for integer, label, and binary representations. Genetic and mutation operators invoked within custom algorithms must support the target encoding configuration.

## Reference Data and Quality Indicators

`problem.optimum` provides a lazily generated and cached tensor of reference points in objective space; `problem.pf` provides an optional representation suitable for plotting. Built-in multimodal problems provide decision-space reference Pareto sets via `problem.ps`; this attribute is not guaranteed for arbitrary problem formulations.

```python
reference_objectives = problem.optimum
drawable_front = problem.pf
reference_decisions = problem.ps  # Multimodal problems only.

```

These properties serve distinct analytical purposes: objective-space indicators consume `optimum`, decision-space indicators consume `ps`, whereas `pf` serves exclusively as a visual curve, meshgrid, or `None`. If a custom multimodal problem does not define `ps`, generate an independent reference set and persist it alongside experimental artifacts.

## Package Resource Access

Underlying benchmark dataset files reside in the `ascendmoea/vendor` resource directory and are resolved internally relative to the installed package path. Inspect packaged data assets using the Python standard library:

```bash
python - <<'PY'
from importlib.resources import files

root = files("ascendmoea").joinpath("vendor")
print(root)
PY

```

Custom problem implementations must not rely on working-directory-relative paths or hardcode absolute filesystem paths into data-loading logic.
