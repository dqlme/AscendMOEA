"""Device selection, precision policy, seeding, and timing synchronization."""

from __future__ import annotations

import os
import random
import warnings

os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")

import numpy as np
import torch

_TORCH_NPU_LOADED = False
_TORCH_NPU_ERROR: Exception | None = None
_CPU_INTEROP_CONFIGURED = False


def _load_torch_npu() -> bool:
    """Load the optional Ascend backend only when it is explicitly needed."""
    global _TORCH_NPU_LOADED, _TORCH_NPU_ERROR
    if _TORCH_NPU_LOADED:
        return hasattr(torch, "npu")
    try:
        import torch_npu  # noqa: F401
    except Exception as exc:
        _TORCH_NPU_ERROR = exc
        return False
    _TORCH_NPU_LOADED = True
    return hasattr(torch, "npu")


def npu_is_available() -> bool:
    """Return whether the Ascend PyTorch backend can use an NPU."""
    if not _load_torch_npu():
        return False
    try:
        return bool(torch.npu.is_available())
    except Exception:
        return False


def select_device(requested: str | None = None) -> torch.device:
    """Resolve an explicit or environment-configured execution device.

    Args:
        requested: Device string such as ``"cpu"``, ``"npu:0"``, ``"cuda:0"``,
            or ``"auto"``. When omitted, ``ASCENDMOEA_DEVICE`` is used.

    Returns:
        The validated PyTorch device.
    """
    req = (requested or os.environ.get("ASCENDMOEA_DEVICE", "auto")).strip().lower()
    if req in {"", "auto"}:
        if npu_is_available():
            return torch.device(f"npu:{os.environ.get('ASCENDMOEA_NPU_INDEX', '0')}")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    if req == "cpu":
        return torch.device("cpu")
    if req.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError("ASCENDMOEA_DEVICE requests CUDA, but CUDA is not available.")
        return torch.device(req)
    if req.startswith("npu"):
        if not npu_is_available():
            msg = "ASCENDMOEA_DEVICE requests NPU, but torch_npu is not available."
            if _TORCH_NPU_ERROR is not None:
                msg = f"{msg} Import error: {_TORCH_NPU_ERROR}"
            raise RuntimeError(msg)
        return torch.device(
            req if ":" in req else f"npu:{os.environ.get('ASCENDMOEA_NPU_INDEX', '0')}"
        )

    return torch.device(req)


def real_dtype_for_device(device: torch.device | str | None) -> torch.dtype:
    """Return the configured real dtype supported by the target device."""
    dev = torch.device("cpu" if device is None else device)
    requested = os.environ.get("ASCENDMOEA_DTYPE", "").strip().lower()
    if requested in {"float32", "fp32", "single"}:
        return torch.float32
    if requested in {"float64", "fp64", "double"}:
        if dev.type == "npu":
            warnings.warn(
                "Ascend NPU does not support all float64 kernels used by AscendMOEA; using float32."
            )
            return torch.float32
        return torch.float64
    return torch.float32 if dev.type == "npu" else torch.float64


def safe_dtype(dtype: torch.dtype | None, device: torch.device | str | None) -> torch.dtype:
    """Map an optional dtype to a representation supported by ``device``."""
    if dtype is None:
        return real_dtype_for_device(device)
    dev = torch.device("cpu" if device is None else device)
    if dev.type == "npu" and dtype == torch.float64:
        return torch.float32
    return dtype


def configure_torch(
    device: torch.device | str,
    cpu_threads: int | None = None,
) -> torch.dtype:
    """Configure precision, CPU threading, and the active accelerator.

    Args:
        device: Target execution device.
        cpu_threads: Intra-op CPU thread count. When omitted, the value comes
            from ``ASCENDMOEA_CPU_THREADS`` and defaults to one.

    Returns:
        The real dtype selected for the device.
    """
    global _CPU_INTEROP_CONFIGURED
    dev = torch.device(device)
    dtype = real_dtype_for_device(dev)
    torch.set_default_dtype(dtype)
    if dev.type == "npu":
        _load_torch_npu()
        torch.npu.set_device(dev)
    elif dev.type == "cpu":
        threads = int(
            os.environ.get("ASCENDMOEA_CPU_THREADS", "1") if cpu_threads is None else cpu_threads
        )
        if threads > 0:
            torch.set_num_threads(threads)
            if not _CPU_INTEROP_CONFIGURED:
                try:
                    torch.set_num_interop_threads(threads)
                except RuntimeError:
                    pass
                _CPU_INTEROP_CONFIGURED = True
    return dtype


def seed_everything(seed: int, device: torch.device | str | None = None) -> None:
    """Seed Python, NumPy, PyTorch, and the selected accelerator backend."""
    value = int(seed)
    random.seed(value)
    np.random.seed(value % (2**32))
    torch.manual_seed(value)
    if device is None:
        return
    dev = torch.device(device)
    if dev.type == "npu" and _load_torch_npu():
        torch.npu.manual_seed_all(value)
    elif dev.type == "cuda" and torch.cuda.is_available():
        torch.cuda.manual_seed_all(value)


def synchronize(device: torch.device | str) -> None:
    """Wait for queued accelerator work to finish for accurate measurements."""
    dev = torch.device(device)
    if dev.type == "npu":
        if not _load_torch_npu():
            raise RuntimeError("Cannot synchronize an unavailable Ascend NPU backend")
        torch.npu.synchronize(dev)
    elif dev.type == "cuda":
        torch.cuda.synchronize(dev)
