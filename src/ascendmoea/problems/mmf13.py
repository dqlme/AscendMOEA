"""Vectorized implementation of the mmf13 problem."""

from __future__ import annotations

from .mmo_base import MMOProblem


class MMF13(MMOProblem):
    """Implement the MMF13 vectorized optimization problem."""

    name = "MMF13"

    def __init__(self, **kwargs):
        """Initialize the instance."""
        super().__init__(15, **kwargs)
