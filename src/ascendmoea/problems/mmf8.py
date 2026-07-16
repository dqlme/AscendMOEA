"""Vectorized implementation of the mmf8 problem."""

from __future__ import annotations

from .mmo_base import MMOProblem


class MMF8(MMOProblem):
    """Implement the MMF8 vectorized optimization problem."""

    name = "MMF8"

    def __init__(self, **kwargs):
        """Initialize the instance."""
        super().__init__(10, **kwargs)
