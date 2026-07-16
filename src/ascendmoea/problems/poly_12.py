"""Vectorized implementation of the poly_12 problem."""

from __future__ import annotations

from .mmo_base import MMOProblem


class Poly12(MMOProblem):
    """Define the Poly12 benchmark with batched tensor evaluation."""

    name = "poly_12"

    def __init__(self, **kwargs):
        """Initialize the suite problem and apply its fixed configuration."""
        super().__init__(70, **kwargs)
