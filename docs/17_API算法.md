# 内置算法 API

所有算法类从 `ascendmoea.algorithms` 导入，也可通过 `create_algorithm(注册名, **参数)`
创建。表中的 `**kwargs` 当前用于传递基类参数 `save`。

算法子模块还公开 `ALGORITHMS`、`register_algorithm`、`get_algorithm`、
`create_algorithm` 和 `list_algorithms`。这些对象与顶层同名入口指向同一注册表；工厂对
未知名称抛出 `KeyError`，注册同名类时默认拒绝覆盖。

## 完整算法表

| 类型 | 注册名 | 类名 | 构造签名 |
|---|---|---|---|
| 通用 | `NSGAII` | `NSGAII` | `(save=-10)` |
| 通用 | `MOEAD` | `MOEAD` | `(agg_type=1, **kwargs)` |
| 通用 | `SPEA2` | `SPEA2` | `(save=-10)` |
| 约束 | `BiCo` | `BiCo` | `(save=-10)` |
| 约束 | `CMOEAD` | `CMOEAD` | `(save=-10)` |
| 约束 | `CCMO` | `CCMO` | `(op_type=1, **kwargs)` |
| 约束 | `CMEGL` | `CMEGL` | `(save=-10)` |
| 约束 | `CMOEACD` | `CMOEACD` | `(e1=1, e2=1, **kwargs)` |
| 约束 | `CMOEAMS` | `CMOEAMS` | `(op_type=1, lam=0.5, **kwargs)` |
| 约束 | `EMCMO` | `EMCMO` | `(save=-10)` |
| 多模态 | `HREA` | `HREA` | `(eps=0.3, p=0.5, **kwargs)` |
| 多模态 | `CMMO` | `CMMO` | `(eta=0.2, tao=0.1, theta=0.1, **kwargs)` |
| 多模态 | `CoMMEA` | `CoMMEA` | `(eps=0.2, **kwargs)` |
| 稀疏 | `SparseEA` | `SparseEA` | `(save=-10)` |
| 稀疏 | `SparseEA2` | `SparseEA2` | `(save=-10)` |
| 稀疏 | `TS-SparseEA` | `TSSparseEA` | `(save=-10, r_eva=0.1, n_group=50)` |
| 大规模 | `LMEA` | `LMEA` | `(save=-10, n_sel=5, n_per=50, n_cor=5, operator_type=1)` |
| 稀疏生成 | `MOEA-PSL` | `MOEAPSL` | `(save=-10)` |

## 通用算法

### `NSGAII`

使用约束感知非支配排序和目标空间拥挤距离进行环境选择。约束为空或恒为零时退化为标准
无约束选择。没有额外参数。

### `MOEAD`

在参考权重邻域内逐子问题更新。`agg_type` 必须为：

| 值 | 聚合函数 |
|---:|---|
| 1 | PBI，惩罚系数为 5 |
| 2 | 加权 Tchebycheff |
| 3 | 归一化加权 Tchebycheff |
| 4 | 修正 Tchebycheff |

参考向量生成器可能调整实际种群数，运行后以 `problem.N` 和结果种群为准。

### `SPEA2`

使用强度适应度、近邻密度和顺序距离截断。大种群的截断可选用 `numba` 编译的 CPU
递推；关闭或不可用时使用语义一致的参考实现。

## 约束算法

### `BiCo`

共同维护主种群和不可行解档案，在约束违反、目标方向和距离截断之间选择候选解。

### `CMOEAD`

把约束违反量加入分解式邻域替换。每个子问题限制替换数量，并在邻域与全局父代之间
随机切换。

### `CCMO`

共同进化一个约束种群和一个忽略约束的辅助种群。`op_type=1` 使用遗传算子，
`op_type=2` 使用差分进化；其他值抛出 `ValueError`。

### `CMEGL`

维护约束、全局和局部三个种群，根据目标统计和可行状态调整全局与局部信息的使用。

### `CMOEACD`

维护探索、可行性利用和多样性增强档案。`e1` 控制探索档案，`e2` 控制可行性档案：

| 值 | 环境选择方式 |
|---:|---|
| 1 | 顺序距离截断 |
| 2 | 拥挤距离 |
| 3 | 参考向量关联 |

两个参数都必须为 1、2 或 3。

### `CMOEAMS`

根据可行解比例和已消耗预算在目标/约束联合阶段与约束支配阶段间切换。
`op_type=1` 使用 GA，`op_type=2` 使用 DE；`lam` 是阶段切换所需可行率，范围 `[0,1]`。

### `EMCMO`

维护约束和无约束辅助种群，并按搜索状态调整候选解传递方式。没有额外构造参数。

## 多模态算法

### `HREA`

使用目标层级和决策空间邻域维护主种群及档案。`eps` 是非负决策邻域参数；`p` 是消耗
一半预算后从档案选择父代的概率，范围 `[0,1]`。

### `CMMO`

共同进化收敛种群和决策多样性种群。`eta` 控制自适应 epsilon 调度起点，`tao` 控制
衰减，`theta` 控制决策空间邻域；当前接口要求三个参数都在 `[0,1]`。

### `CoMMEA`

在普通收敛搜索和局部决策空间搜索之间交换后代。`eps` 是自适应局部收敛半径下界，
必须为正。

## 稀疏与大规模算法

### `SparseEA`

共同进化决策值和布尔激活掩码，变量适应度用于引导稀疏交叉与变异。

### `SparseEA2`

在稀疏掩码搜索上加入按决策值排序的变量分组变异；分组数由实现固定为 4。

### `TSSparseEA`

先执行变量组级掩码搜索，再进入个体级稀疏进化。`r_eva` 是组搜索预算比例，范围
`(0,1]`；`n_group` 是组数，必须为正整数。

### `LMEA`

先进行变量聚类和相关性分析，再交替优化距离变量组和位置变量。参数：

| 参数 | 说明 |
|---|---|
| `n_sel` | 每个变量分析时选取的种群样本数 |
| `n_per` | 每个样本的扰动数 |
| `n_cor` | 每对变量相关性测试次数 |
| `operator_type` | 1 为 GA，2 为 DE |

前三个计数必须为正。

### `MOEA-PSL`

使用生成式隐变量模型学习稀疏 Pareto 集结构，并与掩码搜索结合。没有额外公开参数。

## 运行与返回

所有算法通过 `Algorithm.solve()`、`Workflow` 或 `optimize()` 运行，最终种群从
`OptimizationResult.population` 获取。算法可能在完成整批后略微超过 `max_fe`；应使用
`result.evaluations` 记录实际 FE，不要假设必定精确等于预算。
