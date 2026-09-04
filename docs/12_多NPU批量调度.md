# 多 NPU 批量调度

## 调度边界

AscendMOEA 的 `Workflow` 负责单次运行。多进程并行由外层调度器完成，这样可以让每个
进程拥有独立随机状态、问题评价计数和设备上下文，也便于失败重试和结果隔离。

CPU 和 NPU 使用不同的并发策略：

- CPU：每个任务固定 8 线程，joblib 并行多个独立 seed；
- NPU：每张物理卡启动一个或容量筛选后的少量独立进程；
- 不在线程间共享同一个算法、问题或 NPU 状态。

## 单机八卡示例

`run_one.py` 从环境读取物理卡：

```python
import os

from ascendmoea import create_algorithm, create_problem, optimize

card = int(os.environ["ASCENDMOEA_NPU_INDEX"])
algorithm = create_algorithm(os.environ["ALGORITHM"], save=1)
problem = create_problem(
    os.environ["PROBLEM"],
    n=int(os.environ["POPULATION_SIZE"]),
    max_fe=int(os.environ["MAX_FE"]),
)
result = optimize(
    algorithm,
    problem,
    device=f"npu:{card}",
    seed=int(os.environ["SEED"]),
)
```

为不同卡启动独立进程：

```bash
for card in 0 1 2 3 4 5 6 7; do
  ASCENDMOEA_NPU_INDEX=$card \
  ASCEND_RT_VISIBLE_DEVICES=$card \
  ALGORITHM=NSGAII PROBLEM=DTLZ2 \
  POPULATION_SIZE=1000 MAX_FE=100000 SEED=$((1001 + card)) \
  python run_one.py >"worker_${card}.log" 2>&1 &
done
wait
```

不同 CANN 版本对可见卡环境变量的语义可能不同，应以服务器实际版本验证；程序内部仍要
显式设置 `npu:k`，并在日志中记录逻辑卡与物理卡映射。

## CPU joblib 示例

```python
from joblib import Parallel, delayed


def run_seed(seed):
    return optimize(
        create_algorithm("NSGAII", save=1),
        create_problem("DTLZ2", n=100, max_fe=10_000),
        device="cpu",
        cpu_threads=8,
        seed=seed,
    ).elapsed_seconds


times = Parallel(n_jobs=16, backend="loky")(
    delayed(run_seed)(seed) for seed in range(1001, 1021)
)
```

`n_jobs * cpu_threads` 不应超过经 NUMA 测试确认的有效并发。应给 worker 绑定 CPU 集，
避免跨 NUMA 访问和调度漂移。

## 可靠后台执行

长期队列应使用 systemd、Slurm 或同等作业系统，不要依赖 SSH 会话。systemd 单元应：

- 设置固定工作目录和环境文件；
- `Restart=on-failure`，但限制重启频率；
- 把 stdout/stderr 写入持久日志；
- 使用独立的只读监控 timer；
- 在进度文件长时间不更新时报警，不擅自终止仍有活动的任务；
- 支持从已有 `COMPLETE` 标记跳过已完成任务。

## 队列清单

调度前生成不可变 manifest，每行一个任务：

```json
{"algorithm":"NSGAII","problem":"DTLZ2","seed":1001,"n":1000,"max_fe":100000,"device":"npu:0"}
```

worker 用原子进度文件记录当前位置。队列重启后先验证已有摘要和数据校验，再决定跳过；
不能仅凭目录存在判断完成。

## 监控频率

运行稳定后，外部监控间隔最长 30 分钟；建议服务器本地每 5 分钟做轻量只读检查，远程
控制端每 20 至 30 分钟聚合一次：

- 服务状态和队列心跳；
- worker 完成数、错误数和最旧进度时间；
- 每张 NPU 的进程、利用率、温度和 HBM；
- CPU 利用率、load、PSS、available memory 和 swap；
- 数据盘空间和结果文件可读性；
- 运行时间离群值和 NaN/Inf 指标。

发现代码错误时可以快速停止受影响阶段、修复并重排该阶段，但不相关的正式队列应继续。
