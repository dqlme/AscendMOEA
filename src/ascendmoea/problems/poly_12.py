"""Vectorized implementation of the poly_12 problem."""

from __future__ import annotations

from .mmo_base import MMOProblem


class Poly12(MMOProblem):
    """Implement the Poly12 vectorized optimization problem."""

    name = "poly_12"

    def __init__(self, **kwargs):
        """Initialize the instance."""
        super().__init__(70, **kwargs)
