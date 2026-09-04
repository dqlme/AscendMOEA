# Installation and Ascend Environment

## Software Requirements

Base dependencies:

| Component | Requirement | Purpose |
| --- | --- | --- |
| Python | $\ge$ 3.9 | Runtime and packaging |
| NumPy | 1.24 to 2.x | Data exchange and reference data |
| SciPy | 1.10 to 1.x | Scientific computing and MAT file parsing |
| PyTorch | 2.1 to 2.x | Tensor engine and device dispatching |
| CANN | Matched with driver/firmware | Ascend runtime and operator library |
| `torch_npu` | Matched with PyTorch/CANN | PyTorch NPU backend |

Optional dependencies: `pandas` for experimental data aggregation, `plotly` for interactive visualization, and `numba` for specific CPU acceleration paths.

## CPU Installation

Create an isolated virtual environment in the project root and install:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install .

```

To activate the environment on Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install .

```

## Ascend NPU Installation

Device drivers, firmware, CANN, PyTorch, and `torch_npu` must strictly match Huawei's compatibility matrix. In an existing environment, activate the Python environment and source both CANN environment setup scripts:

```bash
source /path/to/your/python/environment/bin/activate
source /usr/local/Ascend/nnrt/set_env.sh
source /usr/local/Ascend/ascend-toolkit/set_env.sh
cd /path/to/AscendMOEA
python -m pip install -e . --no-deps

```

The `--no-deps` flag prevents generic PyPI resolvers from overriding the device-specific PyTorch build installed on the server. When using `uv`:

```bash
uv pip install -e /path/to/AscendMOEA --no-deps

```

## Installation Verification

```bash
python - <<'PY'
import torch
import ascendmoea

print("version", ascendmoea.__version__)
print("torch", torch.__version__)
print("npu_available", ascendmoea.npu_is_available())
print("algorithms", len(ascendmoea.list_algorithms()))
print("problems", len(ascendmoea.list_problems()))
PY

```

Expected output reflects 18 built-in algorithms and 155 test problems. On NPU systems, verify execution using:

```bash
npu-smi info
python examples/quick_start.py
ASCEND_RT_VISIBLE_DEVICES=0 python examples/cpu_npu_timing.py

```

## Device Selection

Explicit specification is recommended:

```python
result = optimize(algorithm, problem, device="cpu", cpu_threads=8)
result = optimize(algorithm, problem, device="npu:0")

```

Alternatively, set `ASCENDMOEA_DEVICE=cpu` or `ASCENDMOEA_DEVICE=npu:0` and leave `device=None` in code. For reproducible experiments, write device settings explicitly to configuration files to avoid implicit environment variable discrepancies.

## CPU Backend Isolation

Prior to importing PyTorch, the library sets:

```text
TORCH_DEVICE_BACKEND_AUTOLOAD=0

```

This prevents pure CPU workers from inadvertently initializing NPU device contexts if `torch_npu` is installed in the environment. CPU tasks will not appear in the `npu-smi` process list. The device manager imports NPU backends on demand when NPU execution is requested.

## Thread Configuration

The `cpu_threads` parameter configures PyTorch's intra-op compute threads per experimental worker process, not the number of concurrent task repetitions. When specifying an 8-thread CPU baseline run, supply `cpu_threads=8` and configure the matching BLAS/OpenMP environment variables:

```bash
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8

```

If running 16 concurrent runs via `joblib`, total CPU thread demand reaches approximately $16 \times 8$. Thread concurrency must be managed based on physical cores and NUMA topologies to avoid severe oversubscription.

## Common Installation Issues

* `No module named torch_npu`: The CPU backend remains functional; NPU execution requires installing the corresponding backend build.
* NPU unavailable while `npu-smi` is operational: Check CANN environment variables, component version compatibility, and `ASCEND_RT_VISIBLE_DEVICES` settings.
* PyTorch overwritten during package installation: Reinstall the device-specific PyTorch distribution and install AscendMOEA with `--no-deps`.
* Missing MAT files for real-world benchmark problems: Ensure the package is installed as a wheel or standard editable package rather than executing detached `.py` script files.
* CPU worker occupies `NPU0`: Ensure the library is initialized before any top-level `torch` imports, and verify no other module imported `torch_npu` prematurely.
