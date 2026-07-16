"""Vectorized implementation of the mmf3 problem."""

from __future__ import annotations

from .mmo_base import MMOProblem


class MMF3(MMOProblem):
    """Implement the MMF3 vectorized optimization problem."""

    name = "MMF3"

    def __init__(self, **kwargs):
        """Initialize the instance."""
        super().__init__(5, **kwargs)
