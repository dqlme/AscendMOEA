"""Vectorized implementation of the rwmop5 problem."""

from __future__ import annotations

import torch

from .rwmop_base import RWMOPBase


class RWMOP5(RWMOPBase):
    """Disc brake design problem."""

    name = "RWMOP5"
    reference_point = (5.3067, 3.0281682)

    def setting(self):
        """Configure problem dimensions, bounds, encoding, and static data."""
        self.M, self.D = 2, 4
        self.lower = torch.tensor([55.0, 75.0, 1000.0, 11.0], dtype=self.dtype, device=self.device)
        self.upper = torch.tensor([80.0, 110.0, 3000.0, 20.0], dtype=self.dtype, device=self.device)

    def cal_obj(self, x):
        """Evaluate objective values for a batch of decision vectors."""
        x1, x2, x3, x4 = x.unbind(dim=1)
        delta2 = x2**2 - x1**2
        delta3 = x2**3 - x1**3
        return torch.column_stack(
            (4.9e-5 * delta2 * (x4 - 1), 9.82e6 * delta2 / (x3 * x4 * delta3))
        )

    def cal_con(self, x):
        """Evaluate inequality constraints for a batch of decision vectors."""
        x1, x2, x3, x4 = x.unbind(dim=1)
        delta2 = x2**2 - x1**2
        delta3 = x2**3 - x1**3
        return torch.column_stack(
            (
                20 - (x2 - x1),
                x3 / (3.14 * delta2) - 0.4,
                2.22e-3 * x3 * delta3 / delta2**2 - 1,
                900 - 2.66e-2 * x3 * x4 * delta3 / delta2,
            )
        )
