# AscendMOEA

AscendMOEA 是面向华为昇腾 NPU 与通用 CPU 的张量化多目标进化优化库。统一的 PyTorch
数据模型覆盖种群、算法、问题、进化算子、质量指标、同步计时和运行监控；同一套扩展接口
可用于无约束、约束、多模态和稀疏大规模优化。

版本 `0.1.0` 提供 20 个算法、155 个基准或实际问题、19 个公开种群算子和 11 个质量
指标。问题评价与主要种群操作面向完整批次，张量始终保留设备和精度信息。

## 安装

要求 Python 3.9 及以上版本。CPU 环境在项目根目录执行：

```bash
python -m pip install .
```

实验、绘图和开发依赖按需安装：

```bash
python -m pip install ".[experiment]"
python -m pip install ".[plot]"
python -m pip install -e ".[dev,all]"
```

昇腾环境需要预先安装互相兼容的驱动、CANN、PyTorch 和 `torch_npu`。加载 CANN 环境
后，使用 `--no-deps` 可避免替换已适配的 PyTorch：

```bash
source /usr/local/Ascend/nnrt/set_env.sh
source /usr/local/Ascend/ascend-toolkit/set_env.sh
python -m pip install . --no-deps
```

## 快速开始

```python
from ascendmoea import optimize
from ascendmoea.algorithms import NSGAII
from ascendmoea.metrics import hv, igd
from ascendmoea.problems import DTLZ2

problem = DTLZ2(n=100, m=3, max_fe=10_000)
result = optimize(
    NSGAII(save=20),
    problem,
    device="cpu",
    cpu_threads=8,
    seed=42,
)

print("evaluations", result.evaluations)
print("seconds", result.elapsed_seconds)
print("HV", hv(result.population, problem.optimum))
print("IGD", igd(result.population, problem.optimum))
```

在已配置的昇腾环境中，将 `device="cpu"` 改为 `device="npu:0"`。标准工作流会在计时
前后同步设备队列，`elapsed_seconds` 因而覆盖本次运行提交的完整计算。

## 公开模块

- `ascendmoea`：`Population`、`Problem`、`Algorithm`、`Workflow`、监控器和设备接口。
- `ascendmoea.algorithms`：算法类、算法注册表和名称工厂。
- `ascendmoea.problems`：问题类、问题注册表和名称工厂。
- `ascendmoea.operators`：排序、选择、距离、交叉、变异、采样和种群组合。
- `ascendmoea.metrics`：HV、IGD、IGD+、GD、IGDX、PSP 等指标。
- `ascendmoea.experiment`：可复现重复实验、汇总和非参数统计。
- `ascendmoea.plotting`：目标空间、决策空间和指标曲线。

完整接口从 [API 索引](docs/17_API速查.md) 进入。首次使用建议依次阅读
[安装与昇腾环境](docs/02_安装与昇腾环境.md)、
[第一个优化任务](docs/03_第一个优化任务.md) 和
[完整案例集](docs/16_完整案例集.md)。全部文档入口见
[中文文档阅读路线](docs/00_阅读路线.md)。

## 验证

```bash
python -m ruff check src tests examples tools
python -m pytest
python tools/check_release.py
python tools/validate_release.py --device cpu --cpu-threads 1 \
  --output validation-results/cpu.json
```

NPU 环境把验证设备改为 `npu:0`。当前发行构建的环境、检查范围和结果记录在
[发行验证记录](docs/18_发行验证记录.md)。

GitHub CI 在 Python 3.9 和 3.12 上重复执行发布卫生、单元测试、CPU 目录验证和构建；
NPU 门禁必须在具有匹配驱动、CANN、PyTorch 与设备后端的昇腾环境中执行。

## 许可

本项目按 `LICENSE` 中的非商业研究与教育条款发布。算法、基准和数据的必要归属及第三方
条件见 `NOTICE.md`；使用或重新分发前应同时阅读两份文件。
