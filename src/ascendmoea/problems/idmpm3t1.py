"""Vectorized implementation of the idmpm3t1 problem."""

from __future__ import annotations

from .mmo_base import MMOProblem


class IDMPM3T1(MMOProblem):
    """Implement the IDMPM3T1 vectorized optimization problem."""

    name = "IDMPM3T1"

    def __init__(self, **kwargs):
        """Initialize the instance."""
        super().__init__(27, **kwargs)
