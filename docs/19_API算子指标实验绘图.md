# 算子、指标、实验与绘图 API

## 公共种群算子

所有算子从 `ascendmoea.operators` 导入。输入张量应位于同一设备；除明确说明外，输出
保留输入设备和 dtype。

### 支配与排序

```python
domination_matrix(pop_obj: Tensor) -> BoolTensor
domination_count(pop_obj: Tensor) -> Tensor
non_dominated_mask(obj: Tensor) -> BoolTensor
nd_sort(pop_obj: Tensor, pop_con: Tensor | None = None,
        n_sort: int | float = inf) -> tuple[Tensor, int]
```

- `domination_matrix` 输入 `[N,M]`，返回 `[N,N]`；元素 `(i,j)` 表示 `i` 在最小化
  语义下支配 `j`。
- `domination_count` 返回每行被其他个体支配的次数。
- `non_dominated_mask` 只处理目标，不处理约束。
- `nd_sort` 返回每行前沿号和最后已分配前沿。`pop_con > 0` 是违反量；`n_sort` 指定至少
  需要赋有限前沿号的个体数。

矩阵维数错误、目标列为空、约束行数不一致或 `n_sort < 0` 时抛出 `ValueError`。

### 父代选择

```python
tournament_selection(k: int, n: int, *fitness: Tensor) -> LongTensor
roulette_selection(n: int, fitness: Tensor) -> LongTensor
```

`tournament_selection` 返回 `n` 个索引，所有适应度均按越小越优，并按参数顺序进行字典序
比较。`k >= 1`，至少需要一个非空且等长的适应度向量。

`roulette_selection` 对越小越优的有限适应度做非负平移，再按倒数概率采样。它不是直接
接收概率权重的接口；`n` 不能为负。

### 多样性与距离

```python
crowding_distance(pop_obj: Tensor, front_no: Tensor) -> Tensor
pairwise_dist(a: Tensor, b: Tensor) -> Tensor
sequential_distance_truncation(distance: Tensor, delete_count: int) -> BoolTensor
```

`crowding_distance` 在每个有限前沿内计算拥挤距离，边界点为正无穷。`pairwise_dist` 输入
`[A,D]` 与 `[B,D]`，返回欧氏距离 `[A,B]`；大输入使用带消去误差回算的矩阵乘路径。

`sequential_distance_truncation` 接收方阵并返回删除掩码。NPU 上保留距离矩阵构造，顺序
递推迁移到 CPU 执行，以避免大量微小 NPU kernel。安装 `AscendMOEA[accelerator]` 后，
大于内部阈值的 CPU 递推可使用语义一致的编译实现。

```python
sequential_truncation_diagnostics() -> dict[str, object]
reset_sequential_truncation_diagnostics() -> None
warmup_sequential_truncation_accelerator(dtype: torch.dtype | None = None) -> dict[str, object]
```

环境变量 `ASCENDMOEA_COMPILED_TRUNCATION` 可取 `auto`、`off` 或 `required`。

### 交叉与变异

```python
operator_ga(problem, parent_dec, pro_c=1.0, dis_c=20.0,
            pro_m=1.0, dis_m=20.0, half=False) -> Tensor
operator_ga_encoded(problem, parent_dec, pro_c=1.0, dis_c=20.0,
                    pro_m=1.0, dis_m=20.0, half=False) -> Tensor
operator_de(problem, p1, p2, p3, cr=1.0, f=0.5,
            pro_m=1.0, dis_m=20.0) -> Tensor
```

`operator_ga` 根据 `problem.encoding` 对实数、整数、标签、二进制和排列列分别处理。父代
前后两半配对，奇数行的最后一行不参与；`half=True` 每对只返回一个子代。至少需要两行
且列数必须等于 `problem.D`。

`operator_ga_encoded` 是同一公开编码感知语义的独立入口。`genetic_operator` 是
`operator_ga` 的对象别名。

`operator_de` 执行 `p1 + f * (p2 - p3)`、二项交叉和多项式变异。三个父代矩阵必须形状
完全一致且列数为 `problem.D`。`differential_evolution_operator` 是其对象别名。

### 采样与种群

```python
uniform_point(n: int, m: int, device=None) -> tuple[Tensor, int]
merge_pop(*populations: Population) -> Population
```

`uniform_point` 要求 `n >= 1,m >= 2`，返回 simplex-lattice 参考向量和实际向量数；实际数
可能不同于请求数。`merge_pop` 至少接收一个种群，并沿个体维连接三个字段。

## 质量指标

```python
igd(pop, optimum) -> float
gd(pop, optimum) -> float
igdp(pop, optimum) -> float
delta_p(pop, optimum) -> float
dm(pop, optimum) -> float
cpf(pop, optimum) -> float
hv(pop, optimum) -> float
igdx(pop, pos) -> float
coverage_rate(pop, pos) -> float
psp(pop, pos) -> float
feasible_rate(pop, _optimum=None) -> float
```

目标空间指标先使用 `population.best()` 的可行非支配集；决策空间指标使用完整种群的
`population.decs`，以保留多模态解集的覆盖信息。参考数据可为一维单点或二维矩阵；缺失、
为空或维数不匹配时返回 `NaN`。目标空间指标在没有可行解时同样返回 `NaN`，不会猜测
参考数据。

| 注册名 | 函数 | 方向 | 参考空间 |
|---|---|---|---|
| `IGD` | `igd` | 最小 | 目标 |
| `HV` | `hv` | 最大 | 目标 |
| `GD` | `gd` | 最小 | 目标 |
| `IGDp` | `igdp` | 最小 | 目标 |
| `DeltaP` | `delta_p` | 最小 | 目标 |
| `DM` | `dm` | 最大 | 目标 |
| `CPF` | `cpf` | 最大 | 目标 |
| `IGDX` | `igdx` | 最小 | 决策 |
| `CR` | `coverage_rate` | 最大 | 决策 |
| `PSP` | `psp` | 最小 | 决策 |
| `Feasible_rate` | `feasible_rate` | 最大 | 无 |
| `runtime` | `None` | 最小 | 由工作流提供 |

`psp = IGDX / CR`；覆盖率为零时返回正无穷。HV 对两目标和三目标使用切片计算，四目标
及以上使用一百万个设备端蒙特卡洛样本。高维 HV 依赖当前随机状态，重复比较必须固定
seed。

`METRICS` 映射每个注册名到 `(callable, "min" | "max")`，`runtime` 的 callable 为
`None`。

## 重复实验与统计

使用前安装 `AscendMOEA[experiment]`。该模块公开：

```python
RunRecord(algorithm: str, problem: str, run: int, metrics: dict[str, float])

run_single(alg, pro, output_callback=None, should_stop_cb=None, *,
           device=None, seed=None, cpu_threads=None)
    -> tuple[dict[str, float], Algorithm]

run_experiment(alg_factory, pro_factory, n_runs, *,
               device=None, base_seed=None, cpu_threads=None)
    -> list[RunRecord]

records_to_frame(records) -> pandas.DataFrame
summarize(df, metric, style="mean_std") -> pandas.DataFrame
significance_tests(df, metric, test, baseline_alg=None) -> list[dict]
```

`run_single` 使用同步工作流运行一次并计算所有兼容指标。目标指标读取 `pro.optimum`，
IGDX、CR 和 PSP 读取 `pro.ps`；无决策参考集或指标计算不兼容时对应值为 `NaN`。

`run_experiment` 每次都调用两个工厂创建新对象。`base_seed` 非空时第 `r` 次使用
`base_seed + r - 1`；`n_runs < 1` 抛出 `ValueError`。该函数顺序执行，跨重复并行应由
外层进程调度器组织。

`summarize` 的 `style` 可取 `mean`、`mean_std` 或 `median_iqr`。`significance_tests` 的
`test` 可取 `friedman`、`wilcoxon`、`ranksum`，并接受别名 `signrank` 与 `ranksums`。
Friedman 只使用所有算法都存在的配对行且至少需要三个算法；成对检验需要有效
`baseline_alg`。数据不足时返回空列表。

## 绘图

使用前安装 `AscendMOEA[plot]`。所有函数返回 `plotly.graph_objects.Figure`，并把输入
张量分离后迁移到 CPU `float64`。默认字体为 Times New Roman。

```python
draw_obj(obj, pf=None, title="Population (objectives)") -> Figure
draw_problem_obj(problem, pop_obj, title="Population (objectives)",
                 include_pf=True) -> Figure
update_population_trace(fig, pop_obj) -> Figure
draw_dec(dec, title="Population (variables)") -> Figure
draw_metric_curve(fes, values, name) -> Figure
```

- `draw_obj` 只接受 `[N,2]` 或 `[N,3]`。
- `draw_problem_obj` 从 `problem.pf` 和 `problem.optimum` 获取参考几何。
- `update_population_trace` 原地更新名称为 `Population` 的首个 trace。
- `draw_dec` 对一、二、三维分别画点图，高于三维画平行坐标折线。
- `draw_metric_curve` 要求 FE 与指标值元素数相同。

静态 PNG/PDF 输出依赖 Kaleido。函数本身只创建单个坐标图，不生成多子图布局。
