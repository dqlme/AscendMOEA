"""Vectorized implementation of the poly_2 problem."""

from __future__ import annotations

from .mmo_base import MMOProblem


class Poly2(MMOProblem):
    """Implement the Poly2 vectorized optimization problem."""

    name = "poly_2"

    def __init__(self, **kwargs):
        """Initialize the instance."""
        super().__init__(63, **kwargs)
