"""Vectorized implementation of the rwmop9 problem."""

from __future__ import annotations

import math

import torch

from .rwmop_base import RWMOPBase


class RWMOP9(RWMOPBase):
    """Four-bar plane truss problem."""

    name = "RWMOP9"
    reference_point = (3.0485281e3, 4.0e-2)

    def setting(self):
        """Configure problem dimensions, bounds, encoding, and static data."""
        self.M, self.D = 2, 4
        self.lower = torch.tensor(
            [1.0, math.sqrt(2), math.sqrt(2), 1.0], dtype=self.dtype, device=self.device
        )
        self.upper = torch.full((4,), 3.0, dtype=self.dtype, device=self.device)

    def cal_obj(self, x):
        """Evaluate objective values for a batch of decision vectors."""
        x1, x2, x3, x4 = x.unbind(dim=1)
        f1 = 200 * (2 * x1 + math.sqrt(2) * x2 + math.sqrt(2) * x3 + x4)
        f2 = 10 * 200 / 2e5 * (2 / x1 + 2 * math.sqrt(2) / x2 - 2 * math.sqrt(2) / x3 + 2 / x4)
        return torch.column_stack((f1, f2))

    def cal_con(self, x):
        """Evaluate inequality constraints for a batch of decision vectors."""
        return torch.zeros((x.shape[0], 1), dtype=x.dtype, device=x.device)
