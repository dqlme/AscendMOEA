"""Vectorized implementation of the poly_3 problem."""

from __future__ import annotations

from .mmo_base import MMOProblem


class Poly3(MMOProblem):
    """Implement the Poly3 vectorized optimization problem."""

    name = "poly_3"

    def __init__(self, **kwargs):
        """Initialize the instance."""
        super().__init__(64, **kwargs)
