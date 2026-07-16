"""Vectorized implementation of the rwmop10 problem."""

from __future__ import annotations

import math

import torch

from .rwmop_base import RWMOPBase


class RWMOP10(RWMOPBase):
    """Two-bar plane truss problem."""

    name = "RWMOP10"
    reference_point = (1.8704862e2, 6.7710178e-5)

    def setting(self):
        """Configure problem dimensions, bounds, encoding, and static data."""
        self.M, self.D = 2, 2
        self.lower = torch.tensor([0.1, 0.5], dtype=self.dtype, device=self.device)
        self.upper = torch.tensor([2.0, 2.5], dtype=self.dtype, device=self.device)

    def cal_obj(self, x):
        """Evaluate objective values for a batch of decision vectors."""
        x1, x2 = x.unbind(dim=1)
        f1 = 2 * 0.283 * 100 * x2 * torch.sqrt(1 + x1**2)
        f2 = (
            0.283
            * 100
            * (1 + x1**2) ** 1.5
            * torch.sqrt(1 + x1**4)
            / (2 * math.sqrt(2) * 3e7 * x1**2 * x2)
        )
        return torch.column_stack((f1, f2))

    def cal_con(self, x):
        """Evaluate inequality constraints for a batch of decision vectors."""
        x1, x2 = x.unbind(dim=1)
        common = 104 * torch.sqrt(1 + x1**2) / (2 * math.sqrt(2) * x1 * x2)
        return torch.column_stack((common * (1 + x1) - 2e4, common * (1 - x1) - 2e4))
