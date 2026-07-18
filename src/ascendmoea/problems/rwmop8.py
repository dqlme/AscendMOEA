"""Vectorized implementation of the rwmop8 problem."""

from __future__ import annotations

import torch

from .rwmop_base import RWMOPBase


class RWMOP8(RWMOPBase):
    """Car side-impact design problem."""

    name = "RWMOP8"
    reference_point = (9.2596587e1, 4.0, 1.2699733e1)

    def setting(self):
        """Configure problem dimensions, bounds, encoding, and static data."""
        self.M, self.D = 3, 7
        self.lower = torch.tensor(
            [0.5, 0.45, 0.5, 0.5, 0.875, 0.4, 0.4], dtype=self.dtype, device=self.device
        )
        self.upper = torch.tensor(
            [1.5, 1.35, 1.5, 1.5, 2.625, 1.2, 1.2], dtype=self.dtype, device=self.device
        )

    @staticmethod
    def _terms(x):
        """Unpack decisions and derive shared engineering terms."""
        x1, x2, x3, x4, x5, x6, x7 = x.unbind(dim=1)
        vmbp = 10.58 - 0.674 * x1 * x2 - 0.67275 * x2
        vfd = 16.45 - 0.489 * x3 * x7 - 0.843 * x5 * x6
        return x1, x2, x3, x4, x5, x6, x7, vmbp, vfd

    def cal_obj(self, x):
        """Evaluate objective values for a batch of decision vectors."""
        x1, x2, x3, x4, x5, x6, x7, vmbp, vfd = self._terms(x)
        f1 = 1.98 + 4.9 * x1 * 6.67 * x2 + 6.98 * x3 + 4.01 * x4 + 1.78 * x5 + 1e-5 * x6 + 2.73 * x7
        f2 = 4.72 - 0.5 * x4 - 0.19 * x2 * x3
        return torch.column_stack((f1, f2, 0.5 * (vmbp + vfd)))

    def cal_con(self, x):
        """Evaluate inequality constraints for a batch of decision vectors."""
        x1, x2, x3, x4, x5, x6, x7, vmbp, _ = self._terms(x)
        f2 = 4.72 - 0.5 * x4 - 0.19 * x2 * x3
        return torch.column_stack(
            (
                -1 + 1.16 - 0.3717 * x2 * x4 - 0.0092928 * x3,
                -0.32
                + 0.261
                - 0.0159 * x1 * x2
                - 0.06486 * x1
                - 0.019 * x2 * x7
                + 0.0144 * x2 * x5
                + 0.0154464 * x6,
                -0.32 + 0.74 - 0.61 * x2 - 0.031296 * x3 - 0.031872 * x7 + 0.227 * x2**2,
                -0.32
                + 0.214
                + 0.00817 * x5
                - 0.045195 * x1
                - 0.0135168 * x1
                + 0.03099 * x2 * x6
                - 0.018 * x2 * x7
                + 0.007176 * x3
                + 0.023232 * x3
                - 0.00364 * x5 * x6
                - 0.018 * x2**2,
                -32 + 33.86 + 2.95 * x3 - 5.057 * x1 * x2 - 3.795 * x2 - 3.4431 * x7 + 1.45728,
                -32 + 28.98 + 3.818 * x3 - 4.2 * x1 * x2 + 1.27296 * x6 - 2.68065 * x7,
                -32 + 46.36 - 9.9 * x2 - 4.4505 * x1,
                f2 - 4,
                vmbp - 9.9,
            )
        )
