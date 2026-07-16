"""Vectorized implementation of the poly_8 problem."""

from __future__ import annotations

from .mmo_base import MMOProblem


class Poly8(MMOProblem):
    """Implement the Poly8 vectorized optimization problem."""

    name = "poly_8"

    def __init__(self, **kwargs):
        """Initialize the instance."""
        super().__init__(68, **kwargs)
