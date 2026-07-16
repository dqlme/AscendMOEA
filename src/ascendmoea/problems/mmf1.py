"""Vectorized implementation of the mmf1 problem."""

from __future__ import annotations

from .mmo_base import MMOProblem


class MMF1(MMOProblem):
    """Implement the MMF1 vectorized optimization problem."""

    name = "MMF1"

    def __init__(self, **kwargs):
        """Initialize the instance."""
        super().__init__(1, **kwargs)
