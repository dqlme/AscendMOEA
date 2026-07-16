"""Vectorized implementation of the poly_6 problem."""

from __future__ import annotations

from .mmo_base import MMOProblem


class Poly6(MMOProblem):
    """Implement the Poly6 vectorized optimization problem."""

    name = "poly_6"

    def __init__(self, **kwargs):
        """Initialize the instance."""
        super().__init__(67, **kwargs)
