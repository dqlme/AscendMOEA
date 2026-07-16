"""Vectorized implementation of the mmmop6d problem."""

from __future__ import annotations

from .mmo_base import MMOProblem


class MMMOP6D(MMOProblem):
    """Implement the MMMOP6D vectorized optimization problem."""

    name = "MMMOP6D"

    def __init__(self, **kwargs):
        """Initialize the instance."""
        super().__init__(62, **kwargs)
