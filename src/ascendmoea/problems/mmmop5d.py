"""Vectorized implementation of the mmmop5d problem."""

from __future__ import annotations

from .mmo_base import MMOProblem


class MMMOP5D(MMOProblem):
    """Implement the MMMOP5D vectorized optimization problem."""

    name = "MMMOP5D"

    def __init__(self, **kwargs):
        """Initialize the instance."""
        super().__init__(58, **kwargs)
