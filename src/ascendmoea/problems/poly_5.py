"""Vectorized implementation of the poly_5 problem."""

from __future__ import annotations

from .mmo_base import MMOProblem


class Poly5(MMOProblem):
    """Implement the Poly5 vectorized optimization problem."""

    name = "poly_5"

    def __init__(self, **kwargs):
        """Initialize the instance."""
        super().__init__(66, **kwargs)
