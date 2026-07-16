"""Vectorized implementation of the mmmop1b problem."""

from __future__ import annotations

from .mmo_base import MMOProblem


class MMMOP1B(MMOProblem):
    """Implement the MMMOP1B vectorized optimization problem."""

    name = "MMMOP1B"

    def __init__(self, **kwargs):
        """Initialize the instance."""
        super().__init__(44, **kwargs)
