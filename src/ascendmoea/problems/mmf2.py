"""Vectorized implementation of the mmf2 problem."""

from __future__ import annotations

from .mmo_base import MMOProblem


class MMF2(MMOProblem):
    """Implement the MMF2 vectorized optimization problem."""

    name = "MMF2"

    def __init__(self, **kwargs):
        """Initialize the instance."""
        super().__init__(4, **kwargs)
