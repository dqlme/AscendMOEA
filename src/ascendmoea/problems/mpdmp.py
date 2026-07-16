"""Vectorized implementation of the mpdmp problem."""

from __future__ import annotations

import math

import torch

from ..core import Problem
from ..utils import pairwise_dist


class MPDMP(Problem):
    """Implement the MPDMP vectorized optimization problem."""

    name = "MPDMP"

    def __init__(self, n=100, m=10, max_fe=10000, lower=-100.0, upper=100.0):
        """Initialize the instance."""
        self.bound_lower = float(lower)
        self.bound_upper = float(upper)
        super().__init__(n=n, m=max(int(m), 3), d=2, max_fe=max_fe)

    def setting(self) -> None:
        """Configure problem dimensions, bounds, encoding, and static data."""
        self.M = max(self.M, 3)
        self.D = 2
        self.lower = torch.full((2,), self.bound_lower, dtype=self.dtype, device=self.device)
        self.upper = torch.full((2,), self.bound_upper, dtype=self.dtype, device=self.device)
        indices = torch.arange(1, self.M + 1, dtype=self.dtype, device=self.device)
        if self.M % 2 == 0:
            angles = (2 * indices - 3) * torch.pi / self.M
        else:
            angles = (2 * indices - 2) * torch.pi / self.M
        self.points = torch.column_stack([torch.sin(angles), torch.cos(angles)])

    @Problem.device.setter
    def device(self, value) -> None:
        """Compute device."""
        Problem.device.fset(self, value)
        if hasattr(self, "points"):
            self.points = self.points.to(device=self.device, dtype=self.dtype)

    def cal_obj(self, x: torch.Tensor) -> torch.Tensor:
        """Evaluate objective values for a batch of decision vectors."""
        return pairwise_dist(x, self.points.to(device=x.device, dtype=x.dtype))

    def get_optimum(self, n: int = 1000):
        """Generate objective-space reference points for metrics."""
        side = int(math.ceil(math.sqrt(n)))
        axis = torch.linspace(-1, 1, side, dtype=self.dtype, device=self.device)
        gx, gy = torch.meshgrid(axis, axis, indexing="ij")
        candidates = torch.column_stack([gx.reshape(-1), gy.reshape(-1)])
        polygon = self.points.to(device=self.device, dtype=self.dtype)
        following = torch.roll(polygon, shifts=-1, dims=0)
        edge = following - polygon
        relative = candidates[:, None, :] - polygon[None, :, :]
        cross = edge[None, :, 0] * relative[:, :, 1] - edge[None, :, 1] * relative[:, :, 0]
        inside = torch.all(cross >= 0, dim=1) | torch.all(cross <= 0, dim=1)
        return pairwise_dist(candidates[inside], polygon)

    def get_pf(self):
        """Generate a drawable Pareto-front representation."""
        return self.get_optimum(100) if self.M == 3 else None
