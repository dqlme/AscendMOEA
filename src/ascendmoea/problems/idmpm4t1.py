"""Vectorized implementation of the idmpm4t1 problem."""

from __future__ import annotations

from .mmo_base import MMOProblem


class IDMPM4T1(MMOProblem):
    """Implement the IDMPM4T1 vectorized optimization problem."""

    name = "IDMPM4T1"

    def __init__(self, **kwargs):
        """Initialize the instance."""
        super().__init__(31, **kwargs)
