"""Vectorized implementation of the smop_base problem."""

from __future__ import annotations

import torch

from ..core import Problem
from ..utils import uniform_point


def _shape_linear(x: torch.Tensor, m: int) -> torch.Tensor:
    """Compute shape linear."""
    return torch.fliplr(
        torch.cumprod(
            torch.cat(
                [torch.ones((x.shape[0], 1), dtype=x.dtype, device=x.device), x[:, : m - 1]], dim=1
            ),
            dim=1,
        )
    ) * torch.cat(
        [
            torch.ones((x.shape[0], 1), dtype=x.dtype, device=x.device),
            torch.flip(1 - x[:, : m - 1], dims=[1]),
        ],
        dim=1,
    )


def _shape_smop4(x: torch.Tensor, m: int) -> torch.Tensor:
    """Compute shape smop4."""
    y = x[:, : m - 1]
    return torch.fliplr(
        torch.cumprod(
            torch.cat(
                [
                    torch.ones((x.shape[0], 1), dtype=x.dtype, device=x.device),
                    1 - torch.cos(y * torch.pi / 2),
                ],
                dim=1,
            ),
            dim=1,
        )
    ) * torch.cat(
        [
            torch.ones((x.shape[0], 1), dtype=x.dtype, device=x.device),
            torch.flip(1 - torch.sin(y * torch.pi / 2), dims=[1]),
        ],
        dim=1,
    )


def _shape_spherical(x: torch.Tensor, m: int) -> torch.Tensor:
    """Compute shape spherical."""
    y = x[:, : m - 1]
    return torch.fliplr(
        torch.cumprod(
            torch.cat(
                [
                    torch.ones((x.shape[0], 1), dtype=x.dtype, device=x.device),
                    torch.cos(y * torch.pi / 2),
                ],
                dim=1,
            ),
            dim=1,
        )
    ) * torch.cat(
        [
            torch.ones((x.shape[0], 1), dtype=x.dtype, device=x.device),
            torch.flip(torch.sin(y * torch.pi / 2), dims=[1]),
        ],
        dim=1,
    )


def _g1(x: torch.Tensor, t: float):
    """Compute g1."""
    return (x - t) ** 2


def _g2(x: torch.Tensor, t: float):
    """Compute g2."""
    return 2 * (x - t) ** 2 + torch.sin(2 * torch.pi * (x - t)) ** 2


def _g3(x: torch.Tensor, t: float):
    """Compute g3."""
    return 4 - (x - t) - 4 / torch.exp(100 * (x - t) ** 2)


class SMOP(Problem):
    """Implement the SMOP vectorized optimization problem."""

    def __init__(self, problem_id: int, n=100, m=2, d=100, max_fe=10000, theta: float = 0.1):
        """Initialize the instance."""
        self.problem_id = int(problem_id)
        self.theta = float(theta)
        super().__init__(n=n, m=m, d=d, max_fe=max_fe)
        self.name = f"SMOP{self.problem_id}"

    def setting(self):
        """Configure problem dimensions, bounds, encoding, and static data."""
        if self.M is None:
            self.M = 2
        if self.D is None:
            self.D = 100
        self.lower = torch.cat(
            [
                torch.zeros(self.M - 1, dtype=self.dtype, device=self.device),
                -torch.ones(self.D - self.M + 1, dtype=self.dtype, device=self.device),
            ]
        )
        self.upper = torch.cat(
            [
                torch.ones(self.M - 1, dtype=self.dtype, device=self.device),
                2 * torch.ones(self.D - self.M + 1, dtype=self.dtype, device=self.device),
            ]
        )

    def cal_obj(self, x: torch.Tensor) -> torch.Tensor:
        """Evaluate objective values for a batch of decision vectors."""
        n_tail = self.D - self.M + 1
        k = int(torch.ceil(torch.tensor(self.theta * n_tail)).item())
        tail = x[:, self.M - 1 :]
        if self.problem_id == 1:
            g = torch.sum(_g1(tail[:, :k], torch.pi / 3), dim=1) + torch.sum(
                _g2(tail[:, k:], 0), dim=1
            )
            shape = _shape_linear(x, self.M)
        elif self.problem_id == 2:
            g = torch.sum(_g2(tail[:, :k], torch.pi / 3), dim=1) + torch.sum(
                _g3(tail[:, k:], 0), dim=1
            )
            shape = _shape_linear(x, self.M)
        elif self.problem_id == 3:
            g = torch.sum(_g1(tail[:, :k], torch.pi / 3), dim=1)
            rest = tail[:, k:]
            for start in range(0, rest.shape[1], 10):
                temp = 50 - torch.sum(_g1(rest[:, start : start + 10], 0), dim=1)
                g = g + torch.where(temp < 50, temp, torch.zeros_like(temp))
            shape = _shape_linear(x, self.M)
        elif self.problem_id == 4:
            vals = torch.sort(_g3(tail, 0), dim=1).values
            g = torch.sum(vals[:, : max(1, n_tail - k)], dim=1)
            shape = _shape_smop4(x, self.M)
        elif self.problem_id == 5:
            g = torch.sum(_g1(tail, torch.pi / 3) * _g2(tail, 0), dim=1)
            g = g + torch.abs(
                torch.as_tensor(k, dtype=x.dtype, device=x.device)
                - torch.sum(tail != 0, dim=1).to(x.dtype)
            )
            shape = _shape_smop4(x, self.M)
        elif self.problem_id == 6:
            coefficient = torch.linspace(0, 1, n_tail, dtype=x.dtype, device=x.device).unsqueeze(0)
            raw = (tail - torch.pi / 3) ** 2 + coefficient * torch.sin(
                6 * torch.pi * (tail - torch.pi / 3)
            ) ** 2
            values, order = torch.sort(raw, dim=1, stable=True)
            sorted_tail = torch.gather(tail, 1, order)
            drop = sorted_tail == 0
            drop[:, :k] = False
            g = torch.sum(torch.where(drop, torch.zeros_like(values), values), dim=1)
            shape = _shape_smop4(x, self.M)
        elif self.problem_id == 7:
            first = _g2(tail[:, :k], torch.pi / 3).sum(dim=1)
            rest = tail[:, k:]
            target = torch.cat([tail[:, k + 1 :], tail[:, k : k + 1]], dim=1) * 0.9
            g = first + _g2(rest, target).sum(dim=1)
            shape = _shape_spherical(x, self.M)
        else:
            first = _g3(tail[:, :k], torch.remainder(tail[:, 1 : k + 1] + torch.pi, 2)).sum(dim=1)
            g = first + _g3(tail[:, k:-1], tail[:, k + 1 :] * 0.9).sum(dim=1)
            shape = _shape_spherical(x, self.M)
        return (1 + g.reshape(-1, 1) / n_tail) * shape

    def get_optimum(self, n: int = 1000):
        """Generate objective-space reference points for metrics."""
        w, _ = uniform_point(n, self.M, device=self.device)
        w = w.to(dtype=self.dtype)
        if self.problem_id in (1, 2, 3):
            return w

        if self.problem_id in (7, 8):
            return w / torch.linalg.vector_norm(w, dim=1, keepdim=True)

        c = torch.ones_like(w)
        for j in range(2, self.M + 1):
            factors = 1 - c[:, self.M - j + 1 : self.M - 1]
            product = (
                torch.prod(factors, dim=1)
                if factors.shape[1]
                else torch.ones(w.shape[0], dtype=w.dtype, device=w.device)
            )
            temp = w[:, j - 1] / w[:, 0] * product
            c[:, self.M - j] = (temp**2 - temp + torch.sqrt(2 * temp)) / (temp**2 + 1)
        x = torch.acos(torch.clamp(c, -1.0, 1.0)) * 2 / torch.pi
        return _shape_smop4(x, self.M)

    def get_pf(self):
        """Generate a drawable Pareto-front representation."""
        return self.get_optimum(1000)
