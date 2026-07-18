"""Vectorized implementation of the rwmop2 problem."""

from __future__ import annotations

import math

import torch

from .rwmop_base import RWMOPBase


class RWMOP2(RWMOPBase):
    """Vibrating platform problem."""

    name = "RWMOP2"
    reference_point = (-1.2746083e-3, 3.1825489e2)

    def setting(self):
        """Configure problem dimensions, bounds, encoding, and static data."""
        self.M, self.D = 2, 5
        self.lower = torch.tensor([0.05, 0.2, 0.2, 0.35, 3.0], dtype=self.dtype, device=self.device)
        self.upper = torch.tensor([0.5, 0.5, 0.6, 0.5, 6.0], dtype=self.dtype, device=self.device)

    @staticmethod
    def _two_sum(left, right):
        """Return a floating-point sum and its rounding residual."""
        total = left + right
        right_virtual = total - left
        left_virtual = total - right_virtual
        error = (left - left_virtual) + (right - right_virtual)
        return total, error

    @staticmethod
    def _two_product_scalar(value, multiplier):
        """Return a scalar product and its compensated rounding residual."""
        product = value * multiplier
        split = 4097.0 * value
        high = split - (split - value)
        low = value - high
        error = (high * multiplier - product) + low * multiplier
        return product, error

    @classmethod
    def _mass_sum(cls, d1, d2, d3):
        """Evaluate the cancellation-sensitive layered mass expression."""
        if d1.dtype != torch.float32:
            return 100.0 * d1 + 2770.0 * (d2 - d1) + 7780.0 * (d3 - d2)

        # Preserve the small residual of this cancellation-prone linear form.
        p1, e1 = cls._two_product_scalar(d1, -2670.0)
        p2, e2 = cls._two_product_scalar(d2, -5010.0)
        p3, e3 = cls._two_product_scalar(d3, 7780.0)
        subtotal, e4 = cls._two_sum(p1, p2)
        total, e5 = cls._two_sum(subtotal, p3)
        return total + e1 + e2 + e3 + e4 + e5

    @staticmethod
    def _terms(x):
        """Unpack decisions and derive shared platform physics terms."""
        d1, d2, d3, b, length = x.unbind(dim=1)
        rho3 = 7780.0
        e1, e2 = 1.6, 70.0
        mu = 2 * b * RWMOP2._mass_sum(d1, d2, d3)
        # Keep the rho3 term exactly aligned with the validated reference model.
        ei = (2 * b / 3) * (e1 * d1**3 + e2 * (d2**3 - d1**3) + rho3 * (d3 - d2))
        return d1, d2, d3, b, length, mu, ei

    def cal_obj(self, x):
        """Evaluate objective values for a batch of decision vectors."""
        d1, d2, d3, b, length, mu, ei = self._terms(x)
        f1 = -math.pi / (2 * length) ** 2 * torch.sqrt(torch.abs(ei / mu))
        f2 = 2 * b * length * (500 * d1 + 1500 * (d2 - d1) + 800 * (d3 - d2))
        return torch.column_stack((f1, f2))

    def cal_con(self, x):
        """Evaluate inequality constraints for a batch of decision vectors."""
        d1, d2, d3, b, length, _, _ = self._terms(x)
        mass_sum = self._mass_sum(d1, d2, d3)
        mass_constraint = 2 * b * length * (mass_sum - 1400.0 / (b * length))
        return torch.column_stack(
            (mass_constraint, d1 - d2, d2 - d1 - 0.15, d2 - d3, d3 - d2 - 0.01)
        )
