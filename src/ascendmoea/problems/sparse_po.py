"""Vectorized implementation of the sparse_po problem."""

from __future__ import annotations

import numpy as np
import torch

from ..core import Problem
from .real_world_base import field, load_mat, tensor


class SparsePO(Problem):
    """Implement the SparsePO vectorized optimization problem."""

    name = "Sparse_PO"

    def __init__(self, n=100, max_fe=10000):
        """Initialize the instance."""
        super().__init__(n=n, m=2, d=1000, max_fe=max_fe)

    def setting(self) -> None:
        """Configure problem dimensions, bounds, encoding, and static data."""
        dataset = load_mat("Dataset_PO.mat")["Dataset"]
        data = np.asarray(field(dataset, "data1000"), dtype=float)
        self._yield = np.log(data[:, 1:]) - np.log(data[:, :-1])
        self._risk = np.cov(self._yield)
        self.lower = -torch.ones(self.D, dtype=self.dtype, device=self.device)
        self.upper = torch.ones(self.D, dtype=self.dtype, device=self.device)

    def cal_obj(self, x: torch.Tensor) -> torch.Tensor:
        """Evaluate objective values for a batch of decision vectors."""
        weights = x / torch.clamp(torch.sum(torch.abs(x), dim=1, keepdim=True), min=1)
        risk = tensor(self._risk, x)
        yields = tensor(self._yield, x)
        variance = torch.einsum("ni,ij,nj->n", weights, risk, weights)
        returns = 1 - torch.sum(weights @ yields, dim=1)
        return torch.column_stack([variance, returns])

    def get_optimum(self, n: int = 1000):
        """Generate objective-space reference points for metrics."""
        return super().get_optimum(n)

    def get_pf(self):
        """Generate a drawable Pareto-front representation."""
        return None
