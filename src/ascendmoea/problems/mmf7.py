"""Vectorized implementation of the mmf7 problem."""

from __future__ import annotations

from .mmo_base import MMOProblem


class MMF7(MMOProblem):
    """Implement the MMF7 vectorized optimization problem."""

    name = "MMF7"

    def __init__(self, **kwargs):
        """Initialize the instance."""
        super().__init__(9, **kwargs)
