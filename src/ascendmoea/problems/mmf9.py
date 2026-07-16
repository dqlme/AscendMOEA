"""Vectorized implementation of the mmf9 problem."""

from __future__ import annotations

from .mmo_base import MMOProblem


class MMF9(MMOProblem):
    """Implement the MMF9 vectorized optimization problem."""

    name = "MMF9"

    def __init__(self, **kwargs):
        """Initialize the instance."""
        super().__init__(11, **kwargs)
