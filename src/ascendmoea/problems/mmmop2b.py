"""Vectorized implementation of the mmmop2b problem."""

from __future__ import annotations

from .mmo_base import MMOProblem


class MMMOP2B(MMOProblem):
    """Implement the MMMOP2B vectorized optimization problem."""

    name = "MMMOP2B"

    def __init__(self, **kwargs):
        """Initialize the instance."""
        super().__init__(46, **kwargs)
