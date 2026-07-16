"""Vectorized implementation of the mmf4 problem."""

from __future__ import annotations

from .mmo_base import MMOProblem


class MMF4(MMOProblem):
    """Implement the MMF4 vectorized optimization problem."""

    name = "MMF4"

    def __init__(self, **kwargs):
        """Initialize the instance."""
        super().__init__(6, **kwargs)
