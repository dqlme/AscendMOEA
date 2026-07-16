"""Vectorized implementation of the poly_2 problem."""

from __future__ import annotations

from .mmo_base import MMOProblem


class Poly2(MMOProblem):
    """Define the Poly2 benchmark with batched tensor evaluation."""

    name = "poly_2"

    def __init__(self, **kwargs):
        """Initialize the suite problem and apply its fixed configuration."""
        super().__init__(63, **kwargs)
