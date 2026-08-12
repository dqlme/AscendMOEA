# 核心对象与工作流 API

## 包标识与目录对象

```python
from ascendmoea import ALGORITHMS, PROBLEMS, __version__
```

`__version__` 是当前安装包版本字符串。`ALGORITHMS` 和 `PROBLEMS` 分别是算法与问题的
`ComponentRegistry` 实例；可以按映射读取，但扩展代码应优先使用本页后续的注册函数，
以获得重名保护和装饰器语义。

## `Population`

```python
Population(decs: Tensor, objs: Tensor, cons: Tensor)
```

三个字段必须是二维张量，行数相同且位于同一设备。`decs` 和 `objs` 至少有一列；约束
采用 `cons <= 0` 为可行。非张量输入会转换为张量，目标和约束会迁移到决策张量的设备。

| 成员 | 返回 | 说明 |
|---|---|---|
| `decs` | `[N,D] Tensor` | 决策矩阵 |
| `objs` | `[N,M] Tensor` | 最小化目标矩阵 |
| `cons` | `[N,C] Tensor` | 不等式约束矩阵 |
| `feasible` | `[N] bool Tensor` | 每行所有约束均不大于零 |
| `best()` | `Population` | 可行非支配集；单目标时返回最优个体 |
| `clone()` | `Population` | 克隆三个张量 |
| `detach()` | `Population` | 从自动微分图分离 |
| `to(device, dtype=None)` | `Population` | 整体迁移设备与实数精度 |
| `population[index]` | `Population` | 索引后仍保持二维批次 |
| `len(population)` | `int` | 个体数 |

没有可行个体时 `best()` 返回列宽保持不变的空种群。字段维数或行数不一致时构造函数抛出
`ValueError`。

## `Problem`

```python
Problem(n=100, m=2, d=10, max_fe=10_000)
```

`Problem` 是抽象基类，不能直接实例化。子类必须实现 `setting()`，并实现 `cal_obj()` 或
覆盖完整的 `evaluation()`。参数含义：

| 参数 | 说明 |
|---|---|
| `n` | 请求的种群规模，保存为 `N` |
| `m` | 目标数，保存为 `M`；可传 `None` 由 `setting()` 决定 |
| `d` | 决策维数，保存为 `D`；可传 `None` 由 `setting()` 决定 |
| `max_fe` | 目标函数评价预算 |

构造完成后的主要状态：`N`、`M`、`D`、`max_fe`、`FE`、`device`、`dtype`、`lower`、
`upper` 和 `encoding`。`encoding` 每列取值为 1 实数、2 整数、3 标签、4 二进制或 5
排列。

### 评价方法

```python
problem.initialization(n=None) -> Population
problem.evaluate(decisions) -> Population
problem.evaluation(decisions) -> Population
problem.cal_dec(decisions) -> Tensor
problem.cal_obj(decisions) -> Tensor
problem.cal_con(decisions) -> Tensor
```

`evaluate()` 是公开别名。输入可为 `[D]` 或 `[N,D]`，最终按批次处理；通用流程先迁移
设备和 dtype、限制上下界、修复离散编码，再调用 `cal_dec`、`cal_obj` 和 `cal_con`。
目标必须返回 `[N,M]`，约束必须返回 `[N,C]`，且同一问题实例的 `C` 必须稳定。成功后
`FE` 增加批量行数；输出形状错误时抛出 `ValueError` 且不增加 `FE`。

### 参考数据

| 属性或方法 | 说明 |
|---|---|
| `optimum` | 延迟生成并缓存的目标空间参考点 |
| `pf` | 延迟生成并缓存的可绘制前沿表示，可为 `None` |
| `get_optimum(n=1000)` | 生成目标参考点的覆盖入口 |
| `get_pf()` | 生成绘图前沿的覆盖入口 |
| `estimate_feasible_obj(n=5000)` | 对双目标问题估计可行目标样本 |

多模态内置问题另外提供 `ps` 和 `reference_point`，它们不是所有 `Problem` 子类的通用
成员。

## `Algorithm`

```python
Algorithm(save=-10)
```

子类实现 `_solve(problem, output, should_stop, should_pause)`。`abs(save)` 是算法自身
`result` 最多保留的均匀进度槽位数；零和一都保留最新槽位。每条记录是
`(evaluations, population)`。需要严格保存每个代际时使用 `HistoryMonitor`。

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
```

`solve()` 配置设备、线程和随机状态，重置算法历史及 `problem.FE`，然后运行。该方法返回
同一算法实例。`final_population` 返回最后一条记录；未运行时抛出 `RuntimeError`。
直接 `solve()` 不生成运行时间，需计时时使用 `optimize()` 或 `Workflow`。

## 标准工作流

### `RunConfig`

```python
RunConfig(
    device=None,
    seed=None,
    cpu_threads=None,
    synchronize_timing=True,
)
```

该冻结数据类保存单次运行设置。`device=None` 使用环境选择；CPU 未显式传线程时使用
`ASCENDMOEA_CPU_THREADS`，默认一线程。`synchronize_timing=True` 会在计时起止点同步
加速器队列。

### `Workflow`

```python
Workflow(algorithm, problem, monitors=None, config=None)
workflow.run(output=None, should_stop=None, should_pause=None) -> OptimizationResult
```

`monitors` 可为单个 `Monitor` 或序列。`output` 在每个代际边界调用。`should_stop` 返回真
时协作终止；`should_pause` 返回真时暂停，并继续检查停止条件。异常先传给所有监控器的
`on_error`，随后原样抛出。

### `optimize`

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

这是创建 `RunConfig` 和 `Workflow` 后立即运行的便捷入口。

### `OptimizationResult`

| 成员 | 类型 | 说明 |
|---|---|---|
| `algorithm` | `Algorithm` | 已完成运行的算法实例 |
| `problem` | `Problem` | 已完成运行的问题实例 |
| `device` | `torch.device` | 实际设备 |
| `elapsed_seconds` | `float` | 按配置同步后的墙钟时间 |
| `population` | `Population` | 最终记录种群 |
| `evaluations` | `int` | `problem.FE` |

## 监控器

`Monitor` 定义四个无操作生命周期方法：

```python
on_start(algorithm, problem) -> None
on_generation(algorithm, problem) -> None
on_finish(result) -> None
on_error(algorithm, problem, error) -> None
```

`HistoryMonitor(copy_to_cpu=True)` 在每代克隆种群。`records` 是 `GenerationRecord` 列表，
每项含 `evaluations` 和 `population`。大规模运行应实现流式监控器，避免内存随代数增长。

## 注册表

```python
ComponentRegistry(kind, components=())
registry.register(component=None, *, name=None, replace=False)
registry.create(name, **kwargs)
registry.names() -> tuple[str, ...]
```

注册表实现 `MutableMapping[str, type]`。值必须是类；重复名称默认抛出 `KeyError`，只有
显式 `replace=True` 才替换。算法和问题提供以下绑定入口：

```python
register_algorithm(component=None, *, name=None, replace=False)
register_problem(component=None, *, name=None, replace=False)
get_algorithm(name) -> type[Algorithm]
get_problem(name) -> type[Problem]
create_algorithm(name, **kwargs) -> Algorithm
create_problem(name, **kwargs) -> Problem
list_algorithms() -> tuple[str, ...]
list_problems() -> tuple[str, ...]
```

`get_*` 对未知名称抛出 `KeyError`；`create_*` 把关键字参数传给对应类构造函数；`list_*`
按名称排序，适合配置校验和稳定展示。

## 设备 API

| 签名 | 说明 |
|---|---|
| `select_device(requested=None)` | 解析 `cpu`、`cuda:k`、`npu:k` 或 `auto` |
| `npu_is_available()` | 按需加载后端并检查 NPU 可用性 |
| `configure_torch(device, cpu_threads=None)` | 设置 dtype、CPU 线程或当前加速器 |
| `seed_everything(seed, device=None)` | 固定 Python、NumPy、PyTorch 与设备随机状态 |
| `synchronize(device)` | 等待设备队列完成 |

环境变量包括 `ASCENDMOEA_DEVICE`、`ASCENDMOEA_NPU_INDEX`、
`ASCENDMOEA_CPU_THREADS` 和 `ASCENDMOEA_DTYPE`。NPU 执行使用 `float32`；如果请求
`float64`，库会改用 `float32` 并发出警告。
