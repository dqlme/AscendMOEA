# AscendMOEA Documentation Reading Roadmap

## Choose an Entry Point

| Objective | Recommended Reading |
| --- | --- |
| Run algorithms on CPU or Ascend NPU | [Installation and Ascend Environment](02_installation.md), [Simple End-to-End Example](03_first_optimization.md) |
| Select built-in algorithms and test problems | [Algorithm Directory and Selection](05_algorithm_catalog.md), [Problem Suite and Data Resources](06_problem_library.md) |
| Invoke evolutionary operators standalone | [Tensorized Operators Manual](07_tensor_operators.md) |
| Integrate custom problems | [Custom Problem Development](08_custom_problems.md) |
| Implement new algorithms | [Custom Algorithm Development](09_custom_algorithms.md) |
| Run reproducible experiments or save full populations | [Metrics, Experiments, and Monitoring](10_experiments.md) |
| Benchmark CPU vs. NPU performance | [CPU and NPU Performance Specifications](11_performance.md) |
| Schedule batch jobs across multiple NPUs | [Multi-NPU Batch Scheduling](12_multi_npu.md) |
| Troubleshoot installation, numerical precision, or performance issues | [Troubleshooting Guide](13_troubleshooting.md) |
| Find runnable code examples | [Runnable Examples](14_examples.md), [Public API Index](15_api_index.md) |
| Look up classes, functions, arguments, and return types | [Core and Workflow API](16_core_api.md), [Algorithm API](17_algorithm_api.md), [Problem API](18_problem_api.md), [Operators and Analytics API](19_analysis_api.md) |

## Documentation Perspectives

This documentation divides the platform into four orthogonal usage paths:

1. **Solving Path**: Algorithms, problems, workflows, and final populations.

2. **Extension Path**: Registries, base classes, public operators, and user modules.

3. **Experiment Path**: Random seeds, device synchronization, population histories, metrics, and resource logging.

4. **Execution Path**: Environments, performance, batch scheduling, and troubleshooting.

This structure allows users to jump directly into specific tasks without requiring an upfront understanding of the entire framework.

## Documentation Structure

| Scope | Pages |
| --- | --- |
| Installation, Devices, and Quick Start | 01, 02, 03 |
| Core Objects, Algorithms, and Problems | 04, 05, 06 |
| Public Operators and Extension Development | 07, 08, 09 |
| Metrics, Performance, and Multi-Device Scheduling | 10, 11, 12 |
| Troubleshooting Installation and Execution | 13 |
| Full Examples and API References | 14 to 19 |

## Current Version

This documentation corresponds to AscendMOEA `1.0.0`. Code snippets assume Python 3.9+. All objectives are formulated as minimization problems, and constraints follow the standard convention $g(x) \le 0$. When drawing formal performance conclusions, adhere strictly to the synchronization, warm-up, threading, and replication protocols specified in Chapter 11; single interactive runs must not be used as baseline benchmarks.
