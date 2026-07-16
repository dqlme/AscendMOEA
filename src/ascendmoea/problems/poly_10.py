"""Vectorized implementation of the poly_10 problem."""

from __future__ import annotations

from .mmo_base import MMOProblem


class Poly10(MMOProblem):
    """Implement the Poly10 vectorized optimization problem."""

    name = "poly_10"

    def __init__(self, **kwargs):
        """Initialize the instance."""
        super().__init__(69, **kwargs)
