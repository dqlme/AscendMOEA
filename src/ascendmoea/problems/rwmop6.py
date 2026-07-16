"""Vectorized implementation of the rwmop6 problem."""

from __future__ import annotations

import torch

from ..core import round_half_away_from_zero
from .rwmop_base import RWMOPBase


class RWMOP6(RWMOPBase):
    """Speed reducer design problem."""

    name = "RWMOP6"
    reference_point = (5.9698221e3, 1.3e3)

    def setting(self):
        """Configure problem dimensions, bounds, encoding, and static data."""
        self.M, self.D = 2, 7
        self.lower = torch.tensor(
            [2.6, 0.7, 16.51, 7.3, 7.3, 2.9, 5.0], dtype=self.dtype, device=self.device
        )
        self.upper = torch.tensor(
            [3.6, 0.8, 28.49, 8.3, 8.3, 3.9, 5.5], dtype=self.dtype, device=self.device
        )

    @staticmethod
    def _values(x):
        """Compute values."""
        values = list(x.unbind(dim=1))
        values[2] = round_half_away_from_zero(values[2])
        return values

    def cal_obj(self, x):
        """Evaluate objective values for a batch of decision vectors."""
        x1, x2, x3, x4, x5, x6, x7 = self._values(x)
        f1 = (
            0.7854 * x1 * x2**2 * (10 * x3**2 / 3 + 14.933 * x3 - 43.0934)
            - 1.508 * x1 * (x6**2 + x7**2)
            + 7.477 * (x6**3 + x7**3)
            + 0.7854 * (x4 * x6**2 + x5 * x7**2)
        )
        f2 = torch.sqrt((745 * x4 / (x2 * x3)) ** 2 + 1.69e7) / (0.1 * x6**3)
        return torch.column_stack((f1, f2))

    def cal_con(self, x):
        """Evaluate inequality constraints for a batch of decision vectors."""
        x1, x2, x3, x4, x5, x6, x7 = self._values(x)
        f2 = torch.sqrt((745 * x4 / (x2 * x3)) ** 2 + 1.69e7) / (0.1 * x6**3)
        return torch.column_stack(
            (
                1 / (x1 * x2**2 * x3) - 1 / 27,
                1 / (x1 * x2**2 * x3**2) - 1 / 397.5,
                x4**3 / (x2 * x3 * x6**4) - 1 / 1.93,
                x5**3 / (x2 * x3 * x7**4) - 1 / 1.93,
                x2 * x3 - 40,
                x1 / x2 - 12,
                -x1 / x2 + 5,
                1.9 - x4 + 1.5 * x6,
                1.9 - x5 + 1.1 * x7,
                f2 - 1300,
                torch.sqrt((745 * x5 / (x2 * x3)) ** 2 + 1.575e8) / (0.1 * x7**3) - 850,
            )
        )
