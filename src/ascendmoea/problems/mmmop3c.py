"""Vectorized implementation of the mmmop3c problem."""

from __future__ import annotations

from .mmo_base import MMOProblem


class MMMOP3C(MMOProblem):
    """Implement the MMMOP3C vectorized optimization problem."""

    name = "MMMOP3C"

    def __init__(self, **kwargs):
        """Initialize the instance."""
        super().__init__(49, **kwargs)
