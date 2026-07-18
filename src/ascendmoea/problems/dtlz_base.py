"""Vectorized implementation of the dtlz_base problem."""

from __future__ import annotations

import math

import torch

from ..core import Problem
from ..utils import uniform_point


def _linear_shape(x: torch.Tensor, m: int) -> torch.Tensor:
    """Map leading decision variables to the linear DTLZ1 front shape."""
    left = torch.flip(
        torch.cumprod(
            torch.cat(
                [
                    torch.ones((x.shape[0], 1), dtype=x.dtype, device=x.device),
                    x[:, : m - 1],
                ],
                dim=1,
            ),
            dim=1,
        ),
        dims=[1],
    )
    right = torch.cat(
        [
            torch.ones((x.shape[0], 1), dtype=x.dtype, device=x.device),
            torch.flip(1 - x[:, : m - 1], dims=[1]),
        ],
        dim=1,
    )
    return left * right


def _spherical_shape(theta: torch.Tensor) -> torch.Tensor:
    """Map angular decision variables to a spherical DTLZ front."""
    left = torch.flip(
        torch.cumprod(
            torch.cat(
                [
                    torch.ones((theta.shape[0], 1), dtype=theta.dtype, device=theta.device),
                    torch.cos(theta),
                ],
                dim=1,
            ),
            dim=1,
        ),
        dims=[1],
    )
    right = torch.cat(
        [
            torch.ones((theta.shape[0], 1), dtype=theta.dtype, device=theta.device),
            torch.flip(torch.sin(theta), dims=[1]),
        ],
        dim=1,
    )
    return left * right


class DTLZ(Problem):
    """Define the DTLZ benchmark with batched tensor evaluation."""

    def __init__(
        self,
        problem_id: int,
        n: int = 100,
        m: int = 3,
        d: int | None = None,
        max_fe: int = 10000,
    ):
        """Initialize the suite problem and apply its fixed configuration."""
        self.problem_id = int(problem_id)
        if d is None:
            d = m + (4 if self.problem_id == 1 else 19 if self.problem_id == 7 else 9)
        super().__init__(n=n, m=m, d=d, max_fe=max_fe)
        self.name = f"DTLZ{self.problem_id}"

    def setting(self) -> None:
        """Configure problem dimensions, bounds, encoding, and static data."""
        self.lower = torch.zeros(self.D, dtype=self.dtype, device=self.device)
        self.upper = torch.ones(self.D, dtype=self.dtype, device=self.device)

    def cal_obj(self, x: torch.Tensor) -> torch.Tensor:
        """Evaluate objective values for a batch of decision vectors."""
        tail = x[:, self.M - 1 :]
        if self.problem_id in (1, 3):
            g = 100 * (
                tail.shape[1]
                + torch.sum(
                    (tail - 0.5) ** 2 - torch.cos(20 * torch.pi * (tail - 0.5)),
                    dim=1,
                )
            )
        elif self.problem_id in (2, 4, 5):
            g = torch.sum((tail - 0.5) ** 2, dim=1)
        elif self.problem_id == 6:
            g = torch.sum(tail**0.1, dim=1)
        else:
            g = 1 + 9 * torch.mean(tail, dim=1)

        if self.problem_id == 1:
            return 0.5 * (1 + g).unsqueeze(1) * _linear_shape(x, self.M)
        if self.problem_id == 7:
            first = x[:, : self.M - 1]
            last = (1 + g) * (
                self.M
                - torch.sum(
                    first / (1 + g).unsqueeze(1) * (1 + torch.sin(3 * torch.pi * first)),
                    dim=1,
                )
            )
            return torch.column_stack([first, last])

        angles = x[:, : self.M - 1].clone()
        if self.problem_id == 4:
            angles = angles**100
        elif self.problem_id in (5, 6) and self.M > 2:
            angles[:, 1:] = (1 + 2 * g.unsqueeze(1) * angles[:, 1:]) / (2 + 2 * g.unsqueeze(1))
        return (1 + g).unsqueeze(1) * _spherical_shape(angles * torch.pi / 2)

    def get_optimum(self, n: int = 1000):
        """Generate objective-space reference points for metrics."""
        if self.problem_id == 1:
            weights, _ = uniform_point(n, self.M, device=self.device)
            return weights.to(self.dtype) / 2
        if self.problem_id in (2, 3, 4):
            weights, _ = uniform_point(n, self.M, device=self.device)
            weights = weights.to(self.dtype)
            return weights / torch.linalg.vector_norm(weights, dim=1, keepdim=True)
        if self.problem_id in (5, 6):
            values = torch.linspace(0, 1, n, dtype=self.dtype, device=self.device)
            pair = torch.column_stack([values, torch.flip(values, dims=[0])])
            pair = pair / torch.linalg.vector_norm(pair, dim=1, keepdim=True)
            result = torch.cat([pair[:, :1].expand(-1, self.M - 2), pair], dim=1)
            exponents = torch.tensor(
                [self.M - 2, *range(self.M - 2, -1, -1)],
                dtype=self.dtype,
                device=self.device,
            )
            return result / (math.sqrt(2.0) ** exponents).unsqueeze(0)

        side = int(math.ceil(n ** (1 / max(self.M - 1, 1))))
        axes = [
            torch.linspace(0, 1, side, dtype=self.dtype, device=self.device)
            for _ in range(self.M - 1)
        ]
        grid = torch.cartesian_prod(*axes)
        if grid.ndim == 1:
            grid = grid.unsqueeze(1)
        interval = (0.0, 0.251412, 0.631627, 0.859401)
        median = (interval[1] - interval[0]) / (
            interval[3] - interval[2] + interval[1] - interval[0]
        )
        grid = torch.where(
            grid <= median,
            grid * (interval[1] - interval[0]) / median + interval[0],
            (grid - median) * (interval[3] - interval[2]) / (1 - median) + interval[2],
        )
        last = 2 * (self.M - torch.sum(grid / 2 * (1 + torch.sin(3 * torch.pi * grid)), dim=1))
        return torch.column_stack([grid, last])

    def get_pf(self):
        """Generate a drawable Pareto-front representation."""
        return self.get_optimum(100)
