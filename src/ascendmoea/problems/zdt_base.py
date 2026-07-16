"""Vectorized implementation of the zdt_base problem."""

from __future__ import annotations

import math

import torch

from ..core import Problem
from ..utils import nd_sort


class ZDT(Problem):
    """Implement the ZDT vectorized optimization problem."""

    def __init__(
        self,
        problem_id: int,
        n: int = 100,
        d: int | None = None,
        max_fe: int = 10000,
    ):
        """Initialize the instance."""
        self.problem_id = int(problem_id)
        default_d = 80 if self.problem_id == 5 else (10 if self.problem_id in (4, 6) else 30)
        super().__init__(n=n, m=2, d=default_d if d is None else d, max_fe=max_fe)
        self.name = f"ZDT{self.problem_id}"

    def setting(self) -> None:
        """Configure problem dimensions, bounds, encoding, and static data."""
        self.M = 2
        if self.problem_id == 5:
            self.D = int(math.ceil(max(self.D - 30, 1) / 5) * 5 + 30)
            self.encoding = torch.full((self.D,), 4, dtype=torch.long, device=self.device)
            return
        self.lower = torch.zeros(self.D, dtype=self.dtype, device=self.device)
        self.upper = torch.ones(self.D, dtype=self.dtype, device=self.device)
        if self.problem_id == 4:
            self.lower[1:] = -5
            self.upper[1:] = 5

    def cal_obj(self, x: torch.Tensor) -> torch.Tensor:
        """Evaluate objective values for a batch of decision vectors."""
        if self.problem_id == 5:
            first = torch.sum(x[:, :30], dim=1)
            groups = x[:, 30:].reshape(x.shape[0], -1, 5).sum(dim=2)
            values = torch.where(groups < 5, 2 + groups, torch.ones_like(groups))
            f1 = 1 + first
            return torch.column_stack([f1, torch.sum(values, dim=1) / f1])

        if self.problem_id == 6:
            f1 = 1 - torch.exp(-4 * x[:, 0]) * torch.sin(6 * torch.pi * x[:, 0]) ** 6
            g = 1 + 9 * torch.mean(x[:, 1:], dim=1) ** 0.25
            return torch.column_stack([f1, g * (1 - (f1 / g) ** 2)])

        f1 = x[:, 0]
        if self.problem_id == 4:
            tail = x[:, 1:]
            g = (
                1
                + 10 * (self.D - 1)
                + torch.sum(tail**2 - 10 * torch.cos(4 * torch.pi * tail), dim=1)
            )
        else:
            g = 1 + 9 * torch.mean(x[:, 1:], dim=1)
        ratio = f1 / g
        if self.problem_id in (1, 4):
            h = 1 - torch.sqrt(ratio)
        elif self.problem_id == 2:
            h = 1 - ratio**2
        else:
            h = 1 - torch.sqrt(ratio) - ratio * torch.sin(10 * torch.pi * f1)
        return torch.column_stack([f1, g * h])

    def get_optimum(self, n: int = 1000):
        """Generate objective-space reference points for metrics."""
        if self.problem_id == 5:
            first = torch.arange(1, 32, dtype=self.dtype, device=self.device)
            return torch.column_stack([first, ((self.D - 30) / 5) / first])
        lower = 0.280775 if self.problem_id == 6 else 0.0
        first = torch.linspace(lower, 1, n, dtype=self.dtype, device=self.device)
        if self.problem_id in (1, 4):
            second = 1 - torch.sqrt(first)
        elif self.problem_id in (2, 6):
            second = 1 - first**2
        else:
            second = 1 - torch.sqrt(first) - first * torch.sin(10 * torch.pi * first)
        result = torch.column_stack([first, second])
        if self.problem_id == 3:
            result = result[nd_sort(result)[0] == 1]
        return result

    def get_pf(self):
        """Generate a drawable Pareto-front representation."""
        return self.get_optimum(100)
