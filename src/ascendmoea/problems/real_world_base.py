"""Shared resource loading and repair helpers for real-world problems."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import torch


def _vendor_root() -> Path:
    """Compute vendor root."""
    return Path(__file__).resolve().parents[1] / "vendor" / "real_world_data"


@lru_cache(maxsize=None)
def load_mat(filename: str) -> dict:
    """Load and cache a packaged MATLAB data file by filename."""
    import scipy.io

    return scipy.io.loadmat(_vendor_root() / filename, simplify_cells=True)


def field(value, name: str):
    """Read a named field from a mapping, structured array, or object."""
    if isinstance(value, dict):
        return value[name]
    if isinstance(value, np.ndarray) and value.dtype.names:
        return value[name].item()
    return getattr(value, name)


def cell_matrices(value) -> list[np.ndarray]:
    """Normalize MATLAB cell or stacked data into a list of matrices."""
    array = np.asarray(value)
    if array.dtype == object:
        return [np.asarray(item, dtype=float) for item in array.reshape(-1)]
    if array.ndim == 3:
        return [np.asarray(array[index], dtype=float) for index in range(array.shape[0])]
    return [np.asarray(array, dtype=float)]


def tensor(array, like: torch.Tensor) -> torch.Tensor:
    """Create a tensor with the same device and dtype as another tensor."""
    return torch.as_tensor(array, dtype=like.dtype, device=like.device)


def repair_permutation(x: torch.Tensor) -> torch.Tensor:
    """Repair permutation."""
    expected = torch.arange(1, x.shape[1] + 1, dtype=x.dtype, device=x.device)
    sorted_values, rank = torch.sort(x, dim=1, stable=True)
    invalid = torch.any(sorted_values != expected.unsqueeze(0), dim=1)
    repaired = x.clone()
    repaired[invalid] = rank[invalid].to(x.dtype) + 1
    return repaired
