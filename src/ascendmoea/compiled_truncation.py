"""Optional Numba accelerator for exact sequential distance truncation."""

from __future__ import annotations

import time

import numpy as np
import torch

try:
    from numba import njit
except ImportError:
    njit = None


ACCELERATOR_VERSION = "numba_exact_v1"


if njit is not None:

    @njit(cache=True, nogil=True)
    def _sequential_recurrence(
        sorted_distance: np.ndarray,
        neighbor_order: np.ndarray,
        delete_count: int,
    ) -> np.ndarray:
        """Apply iterative nearest-neighbor deletion to a distance matrix."""
        n = sorted_distance.shape[0]
        active = np.ones(n, dtype=np.bool_)
        cursor = np.zeros(n, dtype=np.int64)
        candidates = np.empty(n, dtype=np.int64)

        for _ in range(min(delete_count, n)):
            minimum = np.inf
            candidate_count = 0
            have_numeric_minimum = False
            for row in range(n):
                if not active[row]:
                    continue
                value = sorted_distance[row, cursor[row]]
                if np.isnan(value):
                    if not have_numeric_minimum:
                        candidates[candidate_count] = row
                        candidate_count += 1
                elif not have_numeric_minimum or value < minimum:
                    minimum = value
                    candidates[0] = row
                    candidate_count = 1
                    have_numeric_minimum = True
                elif value == minimum:
                    candidates[candidate_count] = row
                    candidate_count += 1

            chosen = candidates[0]
            for candidate_position in range(1, candidate_count):
                challenger = candidates[candidate_position]
                chosen_cursor = 0
                challenger_cursor = 0
                while chosen_cursor < n and challenger_cursor < n:
                    while chosen_cursor < n and not active[neighbor_order[chosen, chosen_cursor]]:
                        chosen_cursor += 1
                    while (
                        challenger_cursor < n
                        and not active[neighbor_order[challenger, challenger_cursor]]
                    ):
                        challenger_cursor += 1
                    if chosen_cursor == n or challenger_cursor == n:
                        break
                    chosen_value = sorted_distance[chosen, chosen_cursor]
                    challenger_value = sorted_distance[challenger, challenger_cursor]
                    chosen_nan = np.isnan(chosen_value)
                    challenger_nan = np.isnan(challenger_value)
                    if challenger_nan:
                        if chosen_nan:
                            chosen_cursor += 1
                            challenger_cursor += 1
                            continue
                        break
                    if chosen_nan:
                        chosen = challenger
                        break
                    if challenger_value < chosen_value:
                        chosen = challenger
                        break
                    if challenger_value > chosen_value:
                        break
                    chosen_cursor += 1
                    challenger_cursor += 1

            active[chosen] = False
            for row in range(n):
                if not active[row]:
                    continue
                if neighbor_order[row, cursor[row]] != chosen:
                    continue
                next_cursor = cursor[row] + 1
                while next_cursor < n and not active[neighbor_order[row, next_cursor]]:
                    next_cursor += 1
                cursor[row] = next_cursor

        return np.logical_not(active)

else:
    _sequential_recurrence = None


def is_available() -> bool:
    """Return whether the compiled recurrence backend is importable."""
    return _sequential_recurrence is not None


def sequential_recurrence(
    sorted_distance: torch.Tensor,
    neighbor_order: torch.Tensor,
    delete_count: int,
) -> torch.Tensor | None:
    """Execute the exact CPU recurrence and return its deletion mask."""
    if _sequential_recurrence is None:
        return None
    if sorted_distance.device.type != "cpu" or neighbor_order.device.type != "cpu":
        raise ValueError("compiled truncation recurrence requires CPU tensors")
    values = sorted_distance.contiguous().numpy()
    order = neighbor_order.to(torch.int64).contiguous().numpy()
    deleted = _sequential_recurrence(values, order, int(delete_count))
    return torch.from_numpy(deleted)


def warmup(dtype: torch.dtype) -> dict[str, object]:
    """Compile, execute, and validate the recurrence for a real dtype."""
    if _sequential_recurrence is None:
        return {
            "available": False,
            "version": ACCELERATOR_VERSION,
            "dtype": str(dtype),
            "warmup_time_s": 0.0,
        }
    numpy_dtype = np.float32 if dtype == torch.float32 else np.float64
    values = np.array([[1.0, np.inf], [1.0, np.inf]], dtype=numpy_dtype)
    order = np.array([[1, 0], [0, 1]], dtype=np.int64)
    started = time.perf_counter()
    deleted = _sequential_recurrence(values, order, 1)
    elapsed = time.perf_counter() - started
    if not np.array_equal(deleted, np.array([True, False], dtype=np.bool_)):
        raise RuntimeError("compiled truncation warmup returned an invalid mask")
    return {
        "available": True,
        "version": ACCELERATOR_VERSION,
        "dtype": str(dtype),
        "warmup_time_s": elapsed,
    }
