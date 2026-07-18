# 内置问题 API

## 构造与查询

问题既可直接导入，也可按注册名创建：

```python
from ascendmoea import create_problem, get_problem, list_problems
from ascendmoea.problems import DTLZ2

direct = DTLZ2(n=100, m=3, d=12, max_fe=10_000)
configured = create_problem("DTLZ2", n=100, m=3, d=12, max_fe=10_000)
assert get_problem("DTLZ2") is DTLZ2
print(list_problems())
```

注册表名称区分大小写。名称本身是合法 Python 标识符时，`ascendmoea.problems` 同时导出
对应名称及类名；配置文件应始终保存注册名。

## 问题族构造参数

| 问题族 | 公开参数 | 说明 |
|---|---|---|
| `ZDT1` 至 `ZDT6` | `n=100, d=None, max_fe=10000` | 固定双目标；`d=None` 使用各问题默认维数 |
| `DTLZ1` 至 `DTLZ7` | `n=100, m=3, d=None, max_fe=10000` | `d=None` 按问题编号和目标数推导 |
| `CF1` 至 `CF10` | `n=100, m=2, d=10, max_fe=10000` | 目标数由具体问题定义 |
| `DASCMOP1` 至 `DASCMOP9` | `n=100, d=30, max_fe=10000` | 1 至 6 为双目标，7 至 9 为三目标 |
| `LIRCMOP1` 至 `LIRCMOP14` | `n=100, d=30, max_fe=10000` | 1 至 12 为双目标，13 至 14 为三目标 |
| `SMOP1` 至 `SMOP8` | `n=100, m=2, d=100, max_fe=10000, theta=0.1` | `theta` 控制稀疏度 |
| `LSMOP1` 至 `LSMOP9` | `n=100, m=3, d=None, max_fe=10000, nk=5` | 默认 `d=100*m`，`nk` 为子组件数 |
| 内置多模态族 | `n=100, max_fe=10000` | 目标数、维数、边界和参考数据由注册问题确定 |
| `RWMOP1` 至 `RWMOP10` | `n=100, max_fe=10000` | 工程问题的维数与目标数固定 |
| `Sparse_PM/PO/SR` | `n=100, max_fe=10000` | 数据集规格固定 |

实际组合问题的固定规格：

| 注册名 | 构造签名 | 编码或限制 |
|---|---|---|
| `Sparse_KP` | `(n=100, m=2, d=250, max_fe=10000)` | 二进制；只接受 `M=2,D=250` |
| `MOTSP` | `(n=100, m=2, d=30, max_fe=10000)` | 排列；只接受 `M=2,D=30` |
| `mQAP` | `(n=100, m=2, d=10, max_fe=10000)` | 排列；只接受 `M=2,D=10` |
| `MONRP` | `(n=100, d=100, max_fe=10000)` | 二进制；只接受 `D=100` |
| `MPDMP` | `(n=100, m=10, max_fe=10000, lower=-100, upper=100)` | 实数二维决策；`m` 最小为 3 |

给固定数据问题传入不支持的 `m` 或 `d` 会抛出 `ValueError`，不会自动缩放数据集。

## 全部 155 个注册名

### 通用基准：13

```text
ZDT1, ZDT2, ZDT3, ZDT4, ZDT5, ZDT6,
DTLZ1, DTLZ2, DTLZ3, DTLZ4, DTLZ5, DTLZ6, DTLZ7
```

### 约束基准：33

```text
CF1, CF2, CF3, CF4, CF5, CF6, CF7, CF8, CF9, CF10,
DASCMOP1, DASCMOP2, DASCMOP3, DASCMOP4, DASCMOP5, DASCMOP6,
DASCMOP7, DASCMOP8, DASCMOP9,
LIRCMOP1, LIRCMOP2, LIRCMOP3, LIRCMOP4, LIRCMOP5, LIRCMOP6,
LIRCMOP7, LIRCMOP8, LIRCMOP9, LIRCMOP10, LIRCMOP11, LIRCMOP12,
LIRCMOP13, LIRCMOP14
```

### 稀疏与大规模基准：17

```text
SMOP1, SMOP2, SMOP3, SMOP4, SMOP5, SMOP6, SMOP7, SMOP8,
LSMOP1, LSMOP2, LSMOP3, LSMOP4, LSMOP5, LSMOP6, LSMOP7, LSMOP8, LSMOP9
```

### 多模态基准与应用：74

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

### 实际与组合问题：18

```text
RWMOP1, RWMOP2, RWMOP3, RWMOP4, RWMOP5,
RWMOP6, RWMOP7, RWMOP8, RWMOP9, RWMOP10,
Sparse_KP, Sparse_PM, Sparse_PO, Sparse_SR,
MOTSP, mQAP, MONRP, MPDMP
```

## 编码和评价契约

`problem.encoding` 与决策列一一对应：1 实数、2 整数、3 标签、4 二进制、5 排列。
`initialization()` 按编码生成完整批次；`evaluate()` 会限制边界、修复离散值并返回
`Population`。约束全部使用 `g(x) <= 0`。

```python
population = problem.initialization()
reevaluated = problem.evaluate(population.decs[:8])
assert reevaluated.decs.shape == (8, problem.D)
assert reevaluated.objs.shape == (8, problem.M)
```

问题对象会累计 `FE`。同一个实例不应并发用于多个运行，也不应在独立重复实验间复用。

## 参考数据

| 属性 | 适用范围 | 用途 |
|---|---|---|
| `problem.optimum` | 所有问题 | 目标空间指标参考数据 |
| `problem.pf` | 提供绘图几何的问题 | 绘制前沿，可为 `None` |
| `problem.ps` | 内置多模态问题 | IGDX、CR、PSP 的决策参考集 |
| `problem.reference_point` | 内置多模态问题 | 问题定义给出的目标参考点，可为 `None` |

参考数据会随问题迁移到目标设备。实际问题和部分多模态问题从包内 MAT 资源加载，因此
应通过正常安装或 wheel 使用，不能只复制单个 Python 文件。
