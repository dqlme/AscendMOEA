"""Vectorized implementation of the sparse_kp problem."""

from __future__ import annotations

import numpy as np
import torch

from ..core import Problem
from .real_world_base import load_mat, tensor


class SparseKP(Problem):
    """Define the SparseKP benchmark with batched tensor evaluation."""

    name = "Sparse_KP"

    def __init__(self, n=100, m=2, d=250, max_fe=10000):
        """Initialize the suite problem and apply its fixed configuration."""
        if m != 2 or d != 250:
            raise ValueError("the verified Sparse_KP dataset requires M=2 and D=250")
        super().__init__(n=n, m=m, d=d, max_fe=max_fe)

    def setting(self) -> None:
        """Configure problem dimensions, bounds, encoding, and static data."""
        data = load_mat("Dataset_KP-M2-D250.mat")
        self._profit = np.asarray(data["P"], dtype=float)
        self._weight = np.asarray(data["W"], dtype=float)
        self.encoding = torch.full((self.D,), 4, dtype=torch.long, device=self.device)

    def cal_obj(self, x: torch.Tensor) -> torch.Tensor:
        """Evaluate objective values for a batch of decision vectors."""
        profit = tensor(self._profit, x)
        weight = tensor(self._weight, x)
        result = torch.sum(profit, dim=1).unsqueeze(0) - x @ profit.T
        overload = torch.clamp(x @ weight.T - 0.1 * torch.sum(weight, dim=1).unsqueeze(0), min=0)
        return result + 10 * torch.max(overload, dim=1, keepdim=True).values

    def get_optimum(self, n: int = 1000):
        """Generate objective-space reference points for metrics."""
        return torch.sum(
            torch.as_tensor(self._profit, dtype=self.dtype, device=self.device), dim=1
        ).unsqueeze(0)

    def get_pf(self):
        """Generate a drawable Pareto-front representation."""
        return None
