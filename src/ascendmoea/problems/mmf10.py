"""Vectorized implementation of the mmf10 problem."""

from __future__ import annotations

from .mmo_base import MMOProblem


class MMF10(MMOProblem):
    """Implement the MMF10 vectorized optimization problem."""

    name = "MMF10"

    def __init__(self, **kwargs):
        """Initialize the instance."""
        super().__init__(12, **kwargs)
