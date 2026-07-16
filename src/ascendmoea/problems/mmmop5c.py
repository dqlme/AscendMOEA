"""Vectorized implementation of the mmmop5c problem."""

from __future__ import annotations

from .mmo_base import MMOProblem


class MMMOP5C(MMOProblem):
    """Implement the MMMOP5C vectorized optimization problem."""

    name = "MMMOP5C"

    def __init__(self, **kwargs):
        """Initialize the instance."""
        super().__init__(57, **kwargs)
