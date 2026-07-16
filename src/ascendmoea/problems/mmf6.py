"""Vectorized implementation of the mmf6 problem."""

from __future__ import annotations

from .mmo_base import MMOProblem


class MMF6(MMOProblem):
    """Implement the MMF6 vectorized optimization problem."""

    name = "MMF6"

    def __init__(self, **kwargs):
        """Initialize the instance."""
        super().__init__(8, **kwargs)
