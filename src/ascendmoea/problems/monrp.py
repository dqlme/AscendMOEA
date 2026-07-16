"""Vectorized implementation of the monrp problem."""

from __future__ import annotations

import numpy as np
import torch

from ..core import Problem
from .real_world_base import load_mat, tensor


class MONRP(Problem):
    """Define the MONRP benchmark with batched tensor evaluation."""

    name = "MONRP"

    def __init__(self, n=100, d=100, max_fe=10000):
        """Initialize the suite problem and apply its fixed configuration."""
        if d != 100:
            raise ValueError("the verified MONRP dataset requires D=100")
        super().__init__(n=n, m=2, d=d, max_fe=max_fe)

    def setting(self) -> None:
        """Configure problem dimensions, bounds, encoding, and static data."""
        data = load_mat("MONRP-n100-m100.mat")
        self._cost = np.asarray(data["Cost"], dtype=float).reshape(-1)
        self._value = np.asarray(data["Value"], dtype=float)
        self.encoding = torch.full((self.D,), 4, dtype=torch.long, device=self.device)

    def cal_obj(self, x: torch.Tensor) -> torch.Tensor:
        """Evaluate objective values for a batch of decision vectors."""
        cost = tensor(self._cost, x)
        value = tensor(self._value, x)
        return torch.column_stack(
            [
                x @ cost,
                torch.sum(value) - x @ torch.sum(value, dim=1),
            ]
        )

    def get_optimum(self, n: int = 1000):
        """Generate objective-space reference points for metrics."""
        return torch.tensor(
            [[np.sum(self._cost), np.sum(self._value)]],
            dtype=self.dtype,
            device=self.device,
        )

    def get_pf(self):
        """Generate a drawable Pareto-front representation."""
        return None
