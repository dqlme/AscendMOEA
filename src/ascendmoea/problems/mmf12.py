"""Vectorized implementation of the mmf12 problem."""

from __future__ import annotations

from .mmo_base import MMOProblem


class MMF12(MMOProblem):
    """Implement the MMF12 vectorized optimization problem."""

    name = "MMF12"

    def __init__(self, **kwargs):
        """Initialize the instance."""
        super().__init__(14, **kwargs)
