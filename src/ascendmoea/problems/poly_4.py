"""Vectorized implementation of the poly_4 problem."""

from __future__ import annotations

from .mmo_base import MMOProblem


class Poly4(MMOProblem):
    """Implement the Poly4 vectorized optimization problem."""

    name = "poly_4"

    def __init__(self, **kwargs):
        """Initialize the instance."""
        super().__init__(65, **kwargs)
