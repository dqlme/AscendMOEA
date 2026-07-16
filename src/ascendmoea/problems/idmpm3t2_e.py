"""Vectorized implementation of the idmpm3t2_e problem."""

from __future__ import annotations

from .mmo_base import MMOProblem


class IDMPM3T2E(MMOProblem):
    """Implement the IDMPM3T2E vectorized optimization problem."""

    name = "IDMPM3T2_e"

    def __init__(self, **kwargs):
        """Initialize the instance."""
        super().__init__(40, **kwargs)
