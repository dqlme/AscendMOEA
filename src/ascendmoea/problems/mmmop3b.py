"""Vectorized implementation of the mmmop3b problem."""

from __future__ import annotations

from .mmo_base import MMOProblem


class MMMOP3B(MMOProblem):
    """Implement the MMMOP3B vectorized optimization problem."""

    name = "MMMOP3B"

    def __init__(self, **kwargs):
        """Initialize the instance."""
        super().__init__(48, **kwargs)
