"""Vectorized implementation of the sparse_pm problem."""

from __future__ import annotations

import numpy as np
import torch

from ..core import Population, Problem
from .real_world_base import load_mat


class SparsePM(Problem):
    """Implement the SparsePM vectorized optimization problem."""

    name = "Sparse_PM"

    def __init__(self, n=100, max_fe=10000):
        """Initialize the instance."""
        super().__init__(n=n, m=2, d=100, max_fe=max_fe)

    def setting(self) -> None:
        """Configure problem dimensions, bounds, encoding, and static data."""
        data = load_mat("Dataset_PM-D10000-T50-L100-I5-N100.mat")
        self._transactions = np.asarray(data["Data"], dtype=bool)
        self.encoding = torch.full((self.D,), 4, dtype=torch.long, device=self.device)

    def initialization(self, n: int | None = None) -> Population:
        """Create and evaluate an initial population."""
        n = self.N if n is None else int(n)
        decisions = torch.rand((n, self.D), dtype=self.dtype, device=self.device) < 20 / self.D
        return self.evaluation(decisions.to(self.dtype))

    def cal_obj(self, x: torch.Tensor) -> torch.Tensor:
        """Evaluate objective values for a batch of decision vectors."""
        selected = x.to(torch.bool)
        data = torch.as_tensor(self._transactions, dtype=torch.bool, device=x.device)
        missing = (~data).to(x.dtype) @ selected.T.to(x.dtype)
        matched = missing == 0
        counts = torch.sum(matched, dim=0)
        selected_count = torch.sum(selected, dim=1).to(x.dtype)
        inverse_lengths = 1 / torch.sum(data, dim=1).to(x.dtype)
        occupancy = selected_count * (matched.T.to(x.dtype) @ inverse_lengths)
        occupancy = occupancy / torch.clamp(counts.to(x.dtype), min=1)
        result = torch.column_stack([1 - counts.to(x.dtype) / data.shape[0], 1 - occupancy])
        return torch.where((counts == 0).unsqueeze(1), torch.ones_like(result), result)

    def get_optimum(self, n: int = 1000):
        """Generate objective-space reference points for metrics."""
        return super().get_optimum(n)

    def get_pf(self):
        """Generate a drawable Pareto-front representation."""
        return None
