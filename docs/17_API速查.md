# API 速查

## 顶层接口

```python
from ascendmoea import (
    Algorithm,
    ComponentRegistry,
    GenerationRecord,
    HistoryMonitor,
    Monitor,
    OptimizationResult,
    Population,
    Problem,
    RunConfig,
    Workflow,
    configure_torch,
    create_algorithm,
    create_problem,
    get_algorithm,
    get_problem,
    list_algorithms,
    list_problems,
    npu_is_available,
    optimize,
    register_algorithm,
    register_problem,
    seed_everything,
    select_device,
    synchronize,
)
```

## 工作流

```python
optimize(
    algorithm,
    problem,
    *,
    device=None,
    seed=None,
    cpu_threads=None,
    monitors=None,
    synchronize_timing=True,
) -> OptimizationResult
```

`OptimizationResult` 属性：`algorithm`、`problem`、`device`、`elapsed_seconds`、
`population`、`evaluations`。

## 注册表

```python
register_algorithm(cls=None, *, name=None, replace=False)
register_problem(cls=None, *, name=None, replace=False)
get_algorithm(name) -> type[Algorithm]
get_problem(name) -> type[Problem]
create_algorithm(name, **kwargs) -> Algorithm
create_problem(name, **kwargs) -> Problem
list_algorithms() -> tuple[str, ...]
list_problems() -> tuple[str, ...]
```

## Population

```python
population.decs
population.objs
population.cons
len(population)
population[index]
population.clone()
population.detach()
population.to(device)
population.best()
```

## Problem

```python
problem.N             # population size
problem.M             # objective count
problem.D             # decision dimension
problem.max_fe        # evaluation budget
problem.FE            # consumed evaluations
problem.device
problem.dtype
problem.lower
problem.upper
problem.encoding
problem.initialization(n=None) -> Population
problem.evaluate(decisions) -> Population
problem.cal_dec(decisions) -> Tensor
problem.cal_obj(decisions) -> Tensor
problem.cal_con(decisions) -> Tensor
problem.optimum
problem.pf
```

## Algorithm

```python
algorithm.solve(
    problem,
    output=None,
    should_stop=None,
    should_pause=None,
    *,
    device=None,
    cpu_threads=None,
    seed=None,
) -> Algorithm
algorithm.final_population
algorithm.result
```

推荐通过 `Workflow` 或 `optimize` 运行，以获得同步计时和监控生命周期。

## 公共算子

```python
from ascendmoea.operators import (
    crowding_distance,
    differential_evolution_operator,
    domination_count,
    domination_matrix,
    genetic_operator,
    merge_pop,
    nd_sort,
    non_dominated_mask,
    operator_de,
    operator_ga,
    operator_ga_encoded,
    pairwise_dist,
    reset_sequential_truncation_diagnostics,
    roulette_selection,
    sequential_distance_truncation,
    sequential_truncation_diagnostics,
    tournament_selection,
    uniform_point,
    warmup_sequential_truncation_accelerator,
)
```

## 指标

```python
from ascendmoea.metrics import (
    METRICS,
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

`METRICS` 映射值为 `(callable, direction)`，方向是 `"min"` 或 `"max"`；`runtime`
对应函数为 `None`，由工作流结果提供。

## 绘图

```python
from ascendmoea.plotting import (
    draw_dec,
    draw_metric_curve,
    draw_obj,
    draw_problem_obj,
    update_population_trace,
)
```

绘图是可选依赖，安装 `AscendMOEA[plot]` 后使用。

## 环境变量

| 变量 | 作用 |
|---|---|
| `ASCENDMOEA_DEVICE` | 默认设备，如 `cpu`、`npu:0`、`auto` |
| `ASCENDMOEA_NPU_INDEX` | `npu` 未指定编号时的默认卡 |
| `ASCENDMOEA_CPU_THREADS` | 单进程 CPU intra-op 线程数 |
| `ASCENDMOEA_DTYPE` | `float32` 或 `float64` 精度请求 |
| `TORCH_DEVICE_BACKEND_AUTOLOAD` | CPU 隔离时必须为 `0` |

NPU 对部分 `float64` 算子不支持，设备策略会回退为 `float32` 并发出警告。
