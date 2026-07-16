"""Vectorized implementation of the mmf11 problem."""

from __future__ import annotations

from .mmo_base import MMOProblem


class MMF11(MMOProblem):
    """Implement the MMF11 vectorized optimization problem."""

    name = "MMF11"

    def __init__(self, **kwargs):
        """Initialize the instance."""
        super().__init__(13, **kwargs)
