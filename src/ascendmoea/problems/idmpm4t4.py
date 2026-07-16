"""Vectorized implementation of the idmpm4t4 problem."""

from __future__ import annotations

from .mmo_base import MMOProblem


class IDMPM4T4(MMOProblem):
    """Implement the IDMPM4T4 vectorized optimization problem."""

    name = "IDMPM4T4"

    def __init__(self, **kwargs):
        """Initialize the instance."""
        super().__init__(34, **kwargs)
