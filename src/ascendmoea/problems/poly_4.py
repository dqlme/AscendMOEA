"""Vectorized implementation of the poly_4 problem."""

from __future__ import annotations

from .mmo_base import MMOProblem


class Poly4(MMOProblem):
    """Define the Poly4 benchmark with batched tensor evaluation."""

    name = "poly_4"

    def __init__(self, **kwargs):
        """Initialize the suite problem and apply its fixed configuration."""
        super().__init__(65, **kwargs)
