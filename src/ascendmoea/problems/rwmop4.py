"""Vectorized implementation of the rwmop4 problem."""

from __future__ import annotations

import math

import torch

from .rwmop_base import RWMOPBase


class RWMOP4(RWMOPBase):
    """Welded beam design problem."""

    name = "RWMOP4"
    reference_point = (3.6679325e1, 1.3066667e-2)

    def setting(self):
        """Configure problem dimensions, bounds, encoding, and static data."""
        self.M, self.D = 2, 4
        self.lower = torch.tensor([0.125, 0.1, 0.1, 0.125], dtype=self.dtype, device=self.device)
        self.upper = torch.tensor([5.0, 10.0, 10.0, 5.0], dtype=self.dtype, device=self.device)

    @staticmethod
    def _terms(x):
        """Compute terms."""
        x1, x2, x3, x4 = x.unbind(dim=1)
        p, length, e, g = 6000.0, 14.0, 30e6, 12e6
        pc = (
            4.013
            * e
            * torch.sqrt((x3**2 + x4**6) / 36)
            / length**2
            * (1 - x3 / (2 * length) * math.sqrt(e / (4 * g)))
        )
        sigma = 6 * p * length / (x4 * x3**2)
        j = 2 * math.sqrt(2) * x1 * x2 * (x2**2 / 12 + ((x1 + x3) / 2) ** 2)
        r = torch.sqrt(x2**2 / 4 + ((x1 + x3) / 2) ** 2)
        moment = p * (length + x2 / 2)
        tau1 = p / (math.sqrt(2) * x1 * x2)
        tau2 = moment * r / j
        tau = torch.sqrt(tau1**2 + 2 * tau1 * tau2 * x2 / (2 * r) + tau2**2)
        return x1, x2, x3, x4, pc, sigma, tau

    def cal_obj(self, x):
        """Evaluate objective values for a batch of decision vectors."""
        x1, x2, x3, x4, _, _, _ = self._terms(x)
        f1 = 1.10471 * x1**2 * x2 + 0.04811 * x3 * x4 * (14 + x2)
        f2 = 4 * 6000 * 14**3 / (30e6 * x4 * x3**3)
        return torch.column_stack((f1, f2))

    def cal_con(self, x):
        """Evaluate inequality constraints for a batch of decision vectors."""
        x1, _, _, x4, pc, sigma, tau = self._terms(x)
        return torch.column_stack((tau - 13600, sigma - 30000, x1 - x4, 6000 - pc))
