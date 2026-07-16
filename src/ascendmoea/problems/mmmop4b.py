"""Vectorized implementation of the mmmop4b problem."""

from __future__ import annotations

from .mmo_base import MMOProblem


class MMMOP4B(MMOProblem):
    """Implement the MMMOP4B vectorized optimization problem."""

    name = "MMMOP4B"

    def __init__(self, **kwargs):
        """Initialize the instance."""
        super().__init__(52, **kwargs)
