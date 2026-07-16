"""Vectorized implementation of the mmf14_a problem."""

from __future__ import annotations

from .mmo_base import MMOProblem


class MMF14A(MMOProblem):
    """Implement the MMF14A vectorized optimization problem."""

    name = "MMF14_a"

    def __init__(self, **kwargs):
        """Initialize the instance."""
        super().__init__(17, **kwargs)
