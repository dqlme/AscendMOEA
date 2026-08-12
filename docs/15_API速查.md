# 公共 API 索引

本页列出稳定公开模块，并将完整签名、参数、返回值和异常说明分到四个页面。未在这些页面
或模块 `__all__` 中列出的符号属于内部实现，不承诺跨版本保持路径稳定。

## 核心与执行

[核心与工作流 API](16_API核心与工作流.md) 覆盖：

- `Population`、`Problem`、`Algorithm`；
- `RunConfig`、`Workflow`、`OptimizationResult`；
- `Monitor`、`HistoryMonitor`、`GenerationRecord`；
- 设备选择、线程配置、随机种子和同步；
- `ComponentRegistry` 及算法、问题注册工厂。

## 算法

[算法 API](17_API算法.md) 列出 18 个算法的注册名、类名、构造签名、参数范围和适用任务。
直接导入与配置驱动构造都使用同一注册表：

```python
from ascendmoea.algorithms import NSGAII
from ascendmoea import create_algorithm

direct = NSGAII(save=10)
configured = create_algorithm("NSGAII", save=10)
```

## 问题

[问题 API](18_API问题.md) 列出全部 155 个注册名、各问题族接受的构造参数、编码约定、
参考数据属性和直接导入规则。

## 算子、指标、实验和绘图

[算子与分析 API](19_API算子指标实验绘图.md) 覆盖：

- `ascendmoea.operators` 的 19 个公开名称；
- `ascendmoea.metrics` 的 11 个质量指标和 `METRICS`；
- `ascendmoea.experiment` 的重复实验、汇总和统计检验；
- `ascendmoea.plotting` 的五个绘图入口。

## 顶层导出

`ascendmoea` 顶层稳定导出如下：

```text
ALGORITHMS, PROBLEMS, Algorithm, ComponentRegistry, GenerationRecord,
HistoryMonitor, Monitor, OptimizationResult, Population, Problem, RunConfig,
Workflow, __version__, configure_torch, create_algorithm, create_problem,
get_algorithm, get_problem, list_algorithms, list_problems, npu_is_available,
optimize, register_algorithm, register_problem, seed_everything, select_device,
synchronize
```

算法、问题、算子、指标、实验和绘图符号从各自子模块导入，避免顶层命名空间膨胀。
