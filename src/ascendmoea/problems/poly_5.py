"""Vectorized implementation of the poly_5 problem."""

from __future__ import annotations

from .mmo_base import MMOProblem


class Poly5(MMOProblem):
    """Define the Poly5 benchmark with batched tensor evaluation."""

    name = "poly_5"

    def __init__(self, **kwargs):
        """Initialize the suite problem and apply its fixed configuration."""
        super().__init__(66, **kwargs)
