---
hide-toc: true
hide_ai_links: true
---

{.home-title}
# AscendMOEA Documentation

```{raw} html
<section class="hero hero-en">
  <div class="hero-copy">
    <span class="hero-kicker">TENSORIZED · REPRODUCIBLE · ASCEND-NATIVE</span>
    <h1><span>AscendMOEA</span><small>An Ascend NPU-accelerated Platform for<br>Multi-Objective Evolutionary Optimization</small></h1>
    <p>Tensorized workflows for algorithms, problems, operators, and reproducible experiments.</p>
    <div class="hero-actions">
      <a class="primary" href="00_reading_guide.html">Get started</a>
      <a class="secondary" href="15_api_index.html">API Reference</a>
    </div>
  </div>
  <div class="hero-mark" aria-hidden="true">
    <img src="../_static/ascendmoea-mark.svg" alt="">
  </div>
</section>
```

## Built for Ascend, ready for research

::::{grid} 1 2 2 3
:gutter: 3

:::{grid-item-card} ⚡ Tensorized core
:class-card: feature-card
Batch-first PyTorch operators keep population data, precision, and device placement consistent while limiting avoidable transfers.
:::

:::{grid-item-card} ◈ Ascend NPU acceleration
:class-card: feature-card
Run the same workflow on an Ascend NPU for acceleration or on a CPU for development, debugging, and result validation.
:::

:::{grid-item-card} ◎ Research workflow
:class-card: feature-card
Combine algorithms, problem suites, quality indicators, monitoring, and repeated experiments in one reproducible workflow.
:::

::::

## Quick start

::::{tab-set}

:::{tab-item} Install
```bash
python -m pip install .
```
:::

:::{tab-item} Run
```python
from ascendmoea import optimize
from ascendmoea.algorithms import NSGAII
from ascendmoea.problems import DTLZ2

problem = DTLZ2(n=100, m=3, max_fe=10_000)
result = optimize(NSGAII(save=20), problem, device="cpu", seed=42)
print(result.evaluations, result.elapsed_seconds)
```
:::

:::{tab-item} Ascend NPU
```python
result = optimize(
    NSGAII(save=20),
    problem,
    device="npu:0",
    seed=42,
)
```
:::

::::

```{admonition} Project repository
:class: tip
Source code, issue tracking, and contribution entry point: [dqlme/AscendMOEA](https://github.com/dqlme/AscendMOEA)
```

```{toctree}
:hidden:
:maxdepth: 2

00_reading_guide
01_platform_scope
02_installation
03_first_optimization
04_core_workflow
05_algorithm_catalog
06_problem_library
07_tensor_operators
08_custom_problems
09_custom_algorithms
10_experiments
11_performance
12_multi_npu
13_troubleshooting
14_examples
15_api_index
16_core_api
17_algorithm_api
18_problem_api
19_analysis_api
../api/index
```
