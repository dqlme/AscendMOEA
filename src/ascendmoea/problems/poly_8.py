"""Vectorized implementation of the poly_8 problem."""

from __future__ import annotations

from .mmo_base import MMOProblem


class Poly8(MMOProblem):
    """Define the Poly8 benchmark with batched tensor evaluation."""

    name = "poly_8"

    def __init__(self, **kwargs):
        """Initialize the suite problem and apply its fixed configuration."""
        super().__init__(68, **kwargs)
