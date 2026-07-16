"""Vectorized implementation of the rwmop7 problem."""

from __future__ import annotations

import torch

from .rwmop_base import RWMOPBase


class RWMOP7(RWMOPBase):
    """Gear train design problem."""

    name = "RWMOP7"
    reference_point = (3.4655, 4.5417795e1)

    def setting(self):
        """Configure problem dimensions, bounds, encoding, and static data."""
        self.M, self.D = 2, 4
        self.lower = torch.full((4,), 11.51, dtype=self.dtype, device=self.device)
        self.upper = torch.full((4,), 60.49, dtype=self.dtype, device=self.device)

    def cal_obj(self, x):
        """Evaluate objective values for a batch of decision vectors."""
        x1, x2, x3, x4 = x.unbind(dim=1)
        f1 = torch.abs(6.931 - x3 * x4 / (x1 * x2))
        return torch.column_stack((f1, torch.max(x, dim=1).values))

    def cal_con(self, x):
        """Evaluate inequality constraints for a batch of decision vectors."""
        x1, x2, x3, x4 = x.unbind(dim=1)
        f1 = torch.abs(6.931 - x3 * x4 / (x1 * x2))
        return (f1 / 6.931 - 0.5).unsqueeze(1)
