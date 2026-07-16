"""Vectorized implementation of the mmf15_a problem."""

from __future__ import annotations

from .mmo_base import MMOProblem


class MMF15A(MMOProblem):
    """Implement the MMF15A vectorized optimization problem."""

    name = "MMF15_a"

    def __init__(self, **kwargs):
        """Initialize the instance."""
        super().__init__(19, **kwargs)
