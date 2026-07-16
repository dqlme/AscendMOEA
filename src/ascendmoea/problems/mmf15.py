"""Vectorized implementation of the mmf15 problem."""

from __future__ import annotations

from .mmo_base import MMOProblem


class MMF15(MMOProblem):
    """Implement the MMF15 vectorized optimization problem."""

    name = "MMF15"

    def __init__(self, **kwargs):
        """Initialize the instance."""
        super().__init__(18, **kwargs)
