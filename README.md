# AscendMOEA

AscendMOEA 是面向华为昇腾 NPU 与通用 CPU 的张量化多目标进化优化库。项目以
PyTorch 张量为统一数据模型，提供算法、问题、种群、算子、质量指标、运行工作流和
监控接口，可直接运行内置任务，也可扩展用户算法与批量问题。

当前发布版包含 20 个算法、155 个测试或实际问题，以及非支配排序、拥挤距离、锦标赛
选择、轮盘赌、遗传变异、差分进化、参考向量采样和种群合并等公开算子。

## 安装

CPU 环境：

```bash
python -m pip install .
```

昇腾环境需要先安装与设备驱动匹配的 CANN、PyTorch 和 `torch_npu`，再安装本项目：

```bash
source /usr/local/Ascend/nnrt/set_env.sh
source /usr/local/Ascend/ascend-toolkit/set_env.sh
python -m pip install . --no-deps
```

开发模式：

```bash
python -m pip install -e ".[dev,all]"
```

## 首次运行

```python
from ascendmoea import optimize
from ascendmoea.algorithms import NSGAII
from ascendmoea.problems import DTLZ2

result = optimize(
    NSGAII(save=20),
    DTLZ2(n=100, m=3, max_fe=10_000),
    device="cpu",
    cpu_threads=8,
    seed=20260716,
)

print(result.population.objs)
print(result.elapsed_seconds)
```

NPU 运行只需把 `device` 改为 `"npu:0"`。工作流会在计时起止点同步异步设备队列，
因此 `elapsed_seconds` 包含本次算法提交的完整计算。

## 公开入口

- `ascendmoea.algorithms`：算法类、算法注册和工厂函数。
- `ascendmoea.problems`：问题类、问题注册和工厂函数。
- `ascendmoea.operators`：可独立调用的张量化种群算子。
- `ascendmoea.metrics`：HV、IGD、IGD+、IGDX、GD 等指标。
- `ascendmoea.Workflow`：设备配置、同步计时和监控器组合。
- `ascendmoea.Problem`、`ascendmoea.Algorithm`：扩展基类。

## 文档与示例

中文文档从 [`docs/00_阅读路线.md`](docs/00_阅读路线.md) 开始。可执行示例位于
[`examples`](examples)，发布前完整验证入口为
[`tools/validate_release.py`](tools/validate_release.py)。

当前版本的环境、测试范围和设备验证见
[`docs/18_发行验证记录.md`](docs/18_发行验证记录.md)。

## 发布状态

当前版本为 `0.1.0`。安装包中的实际问题数据随
`ascendmoea/vendor` 一并分发。使用、修改或重新发布前请阅读 `LICENSE` 与
`NOTICE.md`。
