# CPU 与 NPU 性能规范

## 两类性能结论

### 单任务延迟

一次算法或算子独占规定资源时的完成时间。CPU 固定 8 线程，NPU 每卡只放一个被测
worker，其他任务不得共享同一设备。单任务加速比应基于这一口径：

```text
speedup = median_cpu8_seconds / median_npu_seconds
```

### 系统吞吐

服务器在并发运行多个独立重复时，每小时完成多少任务。CPU 可用 joblib 并行多个 8 线程
进程，NPU 可在每卡运行一个或经过容量筛选的多个进程。吞吐提升不能替代单任务加速比。

## 标准计时流程

1. 固定算法、问题、`N`、`max_fe`、seed 和 dtype 策略。
2. 完成一次同形状预热，不把首次编译时间混入稳态时间。
3. 清理或记录缓存状态，但不要在某一设备上额外受益。
4. 在计时前同步 NPU。
5. 只计入算法求解，不计入导入、环境激活和结果压缩。
6. 在计时后再次同步 NPU。
7. 记录完整种群、指标、主机内存和设备内存。
8. 至少 20 次独立重复，使用中位数报告。

标准工作流已经实现第 4 和第 6 步：

```python
result = optimize(
    algorithm,
    problem,
    device="npu:0",
    seed=seed,
    synchronize_timing=True,
)
```

## CPU 线程纪律

CPU 基准应显式设置 8 个 intra-op 线程，并限制相关数学库：

```bash
export ASCENDMOEA_CPU_THREADS=8
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export NUMEXPR_NUM_THREADS=8
```

线程越多并不必然越快。小张量上的线程启动、同步和缓存竞争可能让单线程更快。因此应先
做 1、2、4、8、16 线程筛选；若基准协议选择 CPU8，则固定使用 8 线程，同时报告线程
缩放曲线和选择依据。

## 防止 CPU worker 污染 NPU

CPU 进程必须在任何 PyTorch 后端自动发现之前设置：

```bash
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
export ASCENDMOEA_DEVICE=cpu
```

运行后检查 `npu-smi`：CPU worker 不应出现在任何卡上。否则 CPU 时间会混入 NPU 后端
初始化和 CANN 编译开销，主机内存也会因编译进程池显著增大。

## NPU 同步与数据传输

算子压力测试应先把输入传到 NPU，再同步，然后开始计时。只测试目标算子时，不把
CPU 到 NPU 的传输算入内核时间；端到端算法测试则包含算法内部真实发生的传输。两种口径
必须分别命名：

```python
x_npu = x_cpu.to("npu:0")
synchronize("npu:0")
started = time.perf_counter()
y = operator(x_npu)
synchronize("npu:0")
elapsed = time.perf_counter() - started
```

任何 `.item()`、`.cpu()`、`.numpy()` 都可能触发隐式同步。应在代际边界集中处理标量，
避免代内反复阻塞设备队列。

## 如何选择 N 和 max_fe

`N=100`、`max_fe=10_000` 适合质量对照和基础兼容性，但常常不足以体现 NPU。正式性能
设置应先做规模筛选：

| 级别 | 建议 N | 建议 max_fe | 目的 |
|---|---:|---:|---|
| 质量基线 | 100 | 10,000 | 与参考实现对照 |
| 中等压力 | 500 | 50,000 | 判断并行拐点 |
| 高压力 | 1,000 | 100,000 | 展示大批量吞吐 |
| 算子压力 | 2,000 至 10,000 | 不适用 | 测试 `O(N^2)` 算子和内存边界 |

具体设置必须由预实验决定。若 NPU 与 CPU8 差距不显著，依次检查：张量是否真的在 NPU、
是否存在主机同步、批量是否过小、目标函数是否仍有循环、是否首次编译、是否共享卡、以及
`O(N^2)` 内存是否导致退化。只有代码路径正确后才能提高规模。

## NPU 多任务容量

每卡先用 1 个 worker 建立单任务基线，再测试 2、3、4 个并发。若单任务中位时间相对独占
下降超过预设阈值（建议 5%），延迟基准仍用每卡 1 个；吞吐队列可选择使每小时完成量
最大的并发数，但必须标注共享模式。

## 数值精度约定

CPU 默认 `float64`，NPU 默认 `float32`，因为部分昇腾算子不支持 `float64`。这意味着
“忽略浮点误差的一致”必须通过以下方式判断：

- 公式级输出使用 `atol + rtol * abs(reference)`；
- 排序和选择另行比较前沿号、索引集合和平局处理；
- 算法级比较 20 次指标分布，而不是要求相同 seed 得到逐点相同轨迹；
- 对约束边界附近样本单独做容差敏感性分析。

## 必报数据

- CPU8 与 NPU 每次原始墙钟时间；
- 预热时间和是否计入；
- 单次延迟加速比与系统吞吐比；
- CPU/NPU 峰值主机 PSS、RSS；
- NPU 峰值 HBM；
- 每代 FE、种群和指标；
- 失败、超时、重跑和异常原因；
- 硬件利用率时间序列。
