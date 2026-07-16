"""Vectorized implementation of the mmmop4c problem."""

from __future__ import annotations

from .mmo_base import MMOProblem


class MMMOP4C(MMOProblem):
    """Implement the MMMOP4C vectorized optimization problem."""

    name = "MMMOP4C"

    def __init__(self, **kwargs):
        """Initialize the instance."""
        super().__init__(53, **kwargs)
