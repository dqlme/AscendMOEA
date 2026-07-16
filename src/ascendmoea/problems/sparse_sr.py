"""Vectorized implementation of the sparse_sr problem."""

from __future__ import annotations

import numpy as np
import torch

from ..core import Population, Problem
from .real_world_base import load_mat, tensor


class SparseSR(Problem):
    """Implement the SparseSR vectorized optimization problem."""

    name = "Sparse_SR"

    def __init__(self, n=100, max_fe=10000):
        """Initialize the instance."""
        super().__init__(n=n, m=2, d=1024, max_fe=max_fe)

    def setting(self) -> None:
        """Configure problem dimensions, bounds, encoding, and static data."""
        data = load_mat("Dataset_SR-N1024-M480-K260-sigma0.00.mat")
        self._measurement = np.asarray(data["A"], dtype=float)
        self._observation = np.asarray(data["b"], dtype=float).reshape(-1)
        self._true_signal = np.asarray(data["x_true"], dtype=float).reshape(-1)
        nonzero = self._true_signal[self._true_signal != 0]
        lower = np.floor(np.mean(nonzero) - 3 * np.std(nonzero, ddof=1))
        upper = np.ceil(np.mean(nonzero) + 3 * np.std(nonzero, ddof=1))
        self.lower = torch.full((self.D,), float(lower), dtype=self.dtype, device=self.device)
        self.upper = torch.full((self.D,), float(upper), dtype=self.dtype, device=self.device)
        self._total = float(np.linalg.norm(self._observation))

    def initialization(self, n: int | None = None) -> Population:
        """Create and evaluate an initial population."""
        n = self.N if n is None else int(n)
        counts = torch.randint(1, self.D + 1, (n,), device=self.device)
        order = torch.argsort(torch.rand((n, self.D), dtype=self.dtype, device=self.device), dim=1)
        active = torch.arange(self.D, device=self.device).unsqueeze(0) < counts.unsqueeze(1)
        values = torch.randn((n, self.D), dtype=self.dtype, device=self.device) * active
        values = values / torch.clamp(
            torch.linalg.vector_norm(values, dim=1, keepdim=True), min=torch.finfo(self.dtype).eps
        )
        decisions = torch.zeros_like(values)
        decisions.scatter_(1, order, values)
        return self.evaluation(decisions)

    def cal_obj(self, x: torch.Tensor) -> torch.Tensor:
        """Evaluate objective values for a batch of decision vectors."""
        support = (x != 0).to(x.dtype)
        measurement = tensor(self._measurement, x)
        observation = tensor(self._observation, x)
        residual = support @ measurement.T - observation.unsqueeze(0)
        return torch.column_stack(
            [
                torch.mean(support, dim=1),
                torch.linalg.vector_norm(residual, dim=1) / self._total,
            ]
        )

    def get_optimum(self, n: int = 1000):
        """Generate objective-space reference points for metrics."""
        return super().get_optimum(n)

    def get_pf(self):
        """Generate a drawable Pareto-front representation."""
        return None
