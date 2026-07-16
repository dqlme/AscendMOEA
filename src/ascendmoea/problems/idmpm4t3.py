"""Vectorized implementation of the idmpm4t3 problem."""

from __future__ import annotations

from .mmo_base import MMOProblem


class IDMPM4T3(MMOProblem):
    """Implement the IDMPM4T3 vectorized optimization problem."""

    name = "IDMPM4T3"

    def __init__(self, **kwargs):
        """Initialize the instance."""
        super().__init__(33, **kwargs)
