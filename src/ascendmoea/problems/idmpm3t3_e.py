"""Vectorized implementation of the idmpm3t3_e problem."""

from __future__ import annotations

from .mmo_base import MMOProblem


class IDMPM3T3E(MMOProblem):
    """Implement the IDMPM3T3E vectorized optimization problem."""

    name = "IDMPM3T3_e"

    def __init__(self, **kwargs):
        """Initialize the instance."""
        super().__init__(41, **kwargs)
