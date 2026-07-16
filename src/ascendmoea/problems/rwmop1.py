"""Vectorized implementation of the rwmop1 problem."""

from __future__ import annotations

import math

import torch

from ..core import matlab_round
from .rwmop_base import RWMOPBase


class RWMOP1(RWMOPBase):
    """Pressure vessel problem."""

    name = "RWMOP1"
    reference_point = (3.5964885e5, -7.3303829e3)

    def setting(self):
        """Configure problem dimensions, bounds, encoding, and static data."""
        self.M, self.D = 2, 4
        self.lower = torch.tensor([0.51, 0.51, 10.0, 10.0], dtype=self.dtype, device=self.device)
        self.upper = torch.tensor(
            [99.49, 99.49, 200.0, 200.0], dtype=self.dtype, device=self.device
        )

    def cal_obj(self, x):
        """Evaluate objective values for a batch of decision vectors."""
        x1, x2 = matlab_round(x[:, 0]), matlab_round(x[:, 1])
        x3, x4 = x[:, 2], x[:, 3]
        z1 = 0.0625 * x1
        f1 = 1.7781 * z1 * x3**2 + 0.6224 * z1 * x2 * x4 + 3.1661 * z1**2 * x4 + 19.84 * z1**2 * x3
        f2 = -math.pi * x3**2 * x4 - (4.0 / 3.0) * math.pi * x3**3
        return torch.column_stack((f1, f2))

    def cal_con(self, x):
        """Evaluate inequality constraints for a batch of decision vectors."""
        x1, x2 = matlab_round(x[:, 0]), matlab_round(x[:, 1])
        x3 = x[:, 2]
        z1, z2 = 0.0625 * x1, 0.0625 * x2
        return torch.column_stack((0.00954 * x3 - z2, 0.0193 * x3 - z1))
