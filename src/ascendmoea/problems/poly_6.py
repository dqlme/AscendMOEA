"""Vectorized implementation of the poly_6 problem."""

from __future__ import annotations

from .mmo_base import MMOProblem


class Poly6(MMOProblem):
    """Define the Poly6 benchmark with batched tensor evaluation."""

    name = "poly_6"

    def __init__(self, **kwargs):
        """Initialize the suite problem and apply its fixed configuration."""
        super().__init__(67, **kwargs)
