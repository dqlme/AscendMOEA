"""Vectorized implementation of the mmf1_z problem."""

from __future__ import annotations

from .mmo_base import MMOProblem


class MMF1Z(MMOProblem):
    """Implement the MMF1Z vectorized optimization problem."""

    name = "MMF1_z"

    def __init__(self, **kwargs):
        """Initialize the instance."""
        super().__init__(3, **kwargs)
