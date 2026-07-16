"""Vectorized implementation of the mmf14 problem."""

from __future__ import annotations

from .mmo_base import MMOProblem


class MMF14(MMOProblem):
    """Implement the MMF14 vectorized optimization problem."""

    name = "MMF14"

    def __init__(self, **kwargs):
        """Initialize the instance."""
        super().__init__(16, **kwargs)
