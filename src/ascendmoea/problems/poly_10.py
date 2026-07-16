"""Vectorized implementation of the poly_10 problem."""

from __future__ import annotations

from .mmo_base import MMOProblem


class Poly10(MMOProblem):
    """Define the Poly10 benchmark with batched tensor evaluation."""

    name = "poly_10"

    def __init__(self, **kwargs):
        """Initialize the suite problem and apply its fixed configuration."""
        super().__init__(69, **kwargs)
