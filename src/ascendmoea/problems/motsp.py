"""Vectorized implementation of the motsp problem."""

from __future__ import annotations

import torch

from ..core import Problem
from .real_world_base import cell_matrices, load_mat, repair_permutation, tensor


class MOTSP(Problem):
    """Implement the MOTSP vectorized optimization problem."""

    name = "MOTSP"

    def __init__(self, n=100, m=2, d=30, max_fe=10000):
        """Initialize the instance."""
        if m != 2 or d != 30:
            raise ValueError("the verified MOTSP dataset requires M=2 and D=30")
        super().__init__(n=n, m=m, d=d, max_fe=max_fe)

    def setting(self) -> None:
        """Configure problem dimensions, bounds, encoding, and static data."""
        self._costs = cell_matrices(load_mat("MOTSP-M2-D30-c0.0000.mat")["C"])
        self.encoding = torch.full((self.D,), 5, dtype=torch.long, device=self.device)

    def cal_obj(self, x: torch.Tensor) -> torch.Tensor:
        """Evaluate objective values for a batch of decision vectors."""
        route = repair_permutation(x).to(torch.long) - 1
        following = torch.roll(route, shifts=-1, dims=1)
        values = []
        for cost_array in self._costs:
            cost = tensor(cost_array, x)
            values.append(torch.sum(cost[route, following], dim=1))
        return torch.column_stack(values)

    def get_optimum(self, n: int = 1000):
        """Generate objective-space reference points for metrics."""
        return torch.full((1, self.M), float(self.D), dtype=self.dtype, device=self.device)

    def get_pf(self):
        """Generate a drawable Pareto-front representation."""
        return None
