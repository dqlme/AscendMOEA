"""Vectorized implementation of the mmmop5b problem."""

from __future__ import annotations

from .mmo_base import MMOProblem


class MMMOP5B(MMOProblem):
    """Implement the MMMOP5B vectorized optimization problem."""

    name = "MMMOP5B"

    def __init__(self, **kwargs):
        """Initialize the instance."""
        super().__init__(56, **kwargs)
