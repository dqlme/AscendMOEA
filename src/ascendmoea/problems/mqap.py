"""Vectorized implementation of the mqap problem."""

from __future__ import annotations

import numpy as np
import torch

from ..core import Problem
from .real_world_base import cell_matrices, load_mat, repair_permutation, tensor


class MQAP(Problem):
    """Define the MQAP benchmark with batched tensor evaluation."""

    name = "mQAP"

    def __init__(self, n=100, m=2, d=10, max_fe=10000):
        """Initialize the suite problem and apply its fixed configuration."""
        if m != 2 or d != 10:
            raise ValueError("the verified mQAP dataset requires M=2 and D=10")
        super().__init__(n=n, m=m, d=d, max_fe=max_fe)

    def setting(self) -> None:
        """Configure problem dimensions, bounds, encoding, and static data."""
        data = load_mat("mQAP-M2-D10-c0.0000.mat")
        self._distance = np.asarray(data["a"], dtype=float)
        self._flows = cell_matrices(data["b"])
        self.encoding = torch.full((self.D,), 5, dtype=torch.long, device=self.device)

    def cal_obj(self, x: torch.Tensor) -> torch.Tensor:
        """Evaluate objective values for a batch of decision vectors."""
        assignment = repair_permutation(x).to(torch.long) - 1
        distance = tensor(self._distance, x)
        values = []
        for flow_array in self._flows:
            flow = tensor(flow_array, x)
            permuted = flow[assignment.unsqueeze(2), assignment.unsqueeze(1)]
            values.append(torch.sum(permuted * distance.unsqueeze(0), dim=(1, 2)))
        return torch.column_stack(values)

    def get_optimum(self, n: int = 1000):
        """Generate objective-space reference points for metrics."""
        return torch.full((1, self.M), 10000.0 * self.D**2, dtype=self.dtype, device=self.device)

    def get_pf(self):
        """Generate a drawable Pareto-front representation."""
        return None
