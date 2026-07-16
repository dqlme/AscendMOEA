"""Vectorized implementation of the mmf1_e problem."""

from __future__ import annotations

from .mmo_base import MMOProblem


class MMF1E(MMOProblem):
    """Implement the MMF1E vectorized optimization problem."""

    name = "MMF1_e"

    def __init__(self, **kwargs):
        """Initialize the instance."""
        super().__init__(2, **kwargs)
