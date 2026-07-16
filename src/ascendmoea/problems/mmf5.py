"""Vectorized implementation of the mmf5 problem."""

from __future__ import annotations

from .mmo_base import MMOProblem


class MMF5(MMOProblem):
    """Implement the MMF5 vectorized optimization problem."""

    name = "MMF5"

    def __init__(self, **kwargs):
        """Initialize the instance."""
        super().__init__(7, **kwargs)
