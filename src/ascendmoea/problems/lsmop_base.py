"""Vectorized implementation of the lsmop_base problem."""

from __future__ import annotations

import math

import torch

from ..core import Problem
from ..utils import uniform_point
from .smop_base import _shape_linear, _shape_spherical

_FUNCTIONS = {
    1: ("sphere", "sphere"),
    2: ("griewank", "schwefel"),
    3: ("rastrigin", "rosenbrock"),
    4: ("ackley", "griewank"),
    5: ("sphere", "sphere"),
    6: ("rosenbrock", "schwefel"),
    7: ("ackley", "rosenbrock"),
    8: ("griewank", "sphere"),
    9: ("sphere", "ackley"),
}


def _component_value(block: torch.Tensor, function: str) -> torch.Tensor:
    """Compute component value."""
    if function == "sphere":
        return torch.sum(block**2, dim=2)
    if function == "schwefel":
        return torch.max(torch.abs(block), dim=2).values
    if function == "rastrigin":
        return torch.sum(block**2 - 10 * torch.cos(2 * torch.pi * block) + 10, dim=2)
    if function == "rosenbrock":
        return torch.sum(
            100 * (block[:, :, :-1] ** 2 - block[:, :, 1:]) ** 2 + (block[:, :, :-1] - 1) ** 2,
            dim=2,
        )
    if function == "ackley":
        return (
            20
            - 20 * torch.exp(-0.2 * torch.sqrt(torch.mean(block**2, dim=2)))
            - torch.exp(torch.mean(torch.cos(2 * torch.pi * block), dim=2))
            + math.e
        )
    denominator = torch.sqrt(
        torch.arange(1, block.shape[2] + 1, dtype=block.dtype, device=block.device)
    )
    return (
        torch.sum(block**2, dim=2) / 4000
        - torch.prod(torch.cos(block / denominator.view(1, 1, -1)), dim=2)
        + 1
    )


class LSMOP(Problem):
    """Define the LSMOP benchmark with batched tensor evaluation."""

    def __init__(
        self,
        problem_id: int,
        n=100,
        m=3,
        d=None,
        max_fe=10000,
        nk: int = 5,
    ):
        """Initialize the suite problem and apply its fixed configuration."""
        self.problem_id = int(problem_id)
        self.nk = int(nk)
        self.sublen: list[int] = []
        self.group_offsets: list[int] = []
        super().__init__(n=n, m=m, d=d, max_fe=max_fe)
        self.name = f"LSMOP{self.problem_id}"

    def setting(self) -> None:
        """Configure problem dimensions, bounds, encoding, and static data."""
        if self.D is None:
            self.D = 100 * self.M
        self.lower = torch.zeros(self.D, dtype=self.dtype, device=self.device)
        self.upper = torch.cat(
            [
                torch.ones(self.M - 1, dtype=self.dtype, device=self.device),
                10 * torch.ones(self.D - self.M + 1, dtype=self.dtype, device=self.device),
            ]
        )
        coefficients = [3.8 * 0.1 * (1 - 0.1)]
        for _ in range(1, self.M):
            coefficients.append(3.8 * coefficients[-1] * (1 - coefficients[-1]))
        total = sum(coefficients)
        tail_size = self.D - self.M + 1
        self.sublen = [int(value / total * tail_size / self.nk) for value in coefficients]
        if any(length < 1 for length in self.sublen):
            raise ValueError("LSMOP requires at least one variable per subcomponent")
        self.group_offsets = [0]
        for length in self.sublen:
            self.group_offsets.append(self.group_offsets[-1] + length * self.nk)

    def cal_obj(self, x: torch.Tensor) -> torch.Tensor:
        """Evaluate objective values for a batch of decision vectors."""
        transformed = x.clone()
        one_based = torch.arange(self.M, self.D + 1, dtype=x.dtype, device=x.device)
        if self.problem_id <= 4:
            coefficient = 1 + one_based / self.D
        else:
            coefficient = 1 + torch.cos(one_based / self.D * torch.pi / 2)
        transformed[:, self.M - 1 :] = (
            coefficient.unsqueeze(0) * transformed[:, self.M - 1 :] - 10 * x[:, :1]
        )

        functions = _FUNCTIONS[self.problem_id]
        values = []
        for objective in range(self.M):
            length = self.sublen[objective]
            start = self.M - 1 + self.group_offsets[objective]
            stop = start + length * self.nk
            block = transformed[:, start:stop].reshape(x.shape[0], self.nk, length)
            function = functions[objective % 2]
            value = torch.sum(_component_value(block, function), dim=1)
            values.append(value / (length * self.nk))
        g = torch.column_stack(values)

        if self.problem_id <= 4:
            return (1 + g) * _shape_linear(transformed, self.M)
        if self.problem_id <= 8:
            coupled = torch.cat([g[:, 1:], torch.zeros_like(g[:, :1])], dim=1)
            return (1 + g + coupled) * _shape_spherical(transformed, self.M)

        total_g = 1 + torch.sum(g, dim=1)
        first = transformed[:, : self.M - 1]
        last = (1 + total_g) * (
            self.M
            - torch.sum(
                first / (1 + total_g).unsqueeze(1) * (1 + torch.sin(3 * torch.pi * first)),
                dim=1,
            )
        )
        return torch.column_stack([first, last])

    def get_optimum(self, n: int = 1000):
        """Generate objective-space reference points for metrics."""
        if self.problem_id <= 4:
            weights, _ = uniform_point(n, self.M, device=self.device)
            return weights.to(self.dtype)
        if self.problem_id <= 8:
            weights, _ = uniform_point(n, self.M, device=self.device)
            weights = weights.to(self.dtype)
            return weights / torch.linalg.vector_norm(weights, dim=1, keepdim=True)

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
        return self.get_optimum(1000)
