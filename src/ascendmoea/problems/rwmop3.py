"""Vectorized implementation of the rwmop3 problem."""

from __future__ import annotations

import torch

from .rwmop_base import RWMOPBase


class RWMOP3(RWMOPBase):
    """Two-bar truss design problem."""

    name = "RWMOP3"
    reference_point = (1.0e-1, 1.0e5)

    def setting(self):
        """Configure problem dimensions, bounds, encoding, and static data."""
        self.M, self.D = 2, 3
        self.lower = torch.tensor([1e-5, 1e-5, 1.0], dtype=self.dtype, device=self.device)
        self.upper = torch.tensor([100.0, 100.0, 3.0], dtype=self.dtype, device=self.device)

    def cal_obj(self, x):
        """Evaluate objective values for a batch of decision vectors."""
        x1, x2, x3 = x.unbind(dim=1)
        f1 = x1 * torch.sqrt(16 + x3**2) + x2 * torch.sqrt(1 + x3**2)
        f2 = 20 * torch.sqrt(16 + x3**2) / (x3 * x1)
        return torch.column_stack((f1, f2))

    def cal_con(self, x):
        """Evaluate inequality constraints for a batch of decision vectors."""
        x1, x2, x3 = x.unbind(dim=1)
        f1 = x1 * torch.sqrt(16 + x3**2) + x2 * torch.sqrt(1 + x3**2)
        f2 = 20 * torch.sqrt(16 + x3**2) / (x3 * x1)
        g3 = 80 * torch.sqrt(1 + x3**2) / (x3 * x2) - 1e5
        return torch.column_stack((f1 - 0.1, f2 - 1e5, g3))
