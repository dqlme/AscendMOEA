"""Vectorized implementation of the mmmop6b problem."""

from __future__ import annotations

from .mmo_base import MMOProblem


class MMMOP6B(MMOProblem):
    """Implement the MMMOP6B vectorized optimization problem."""

    name = "MMMOP6B"

    def __init__(self, **kwargs):
        """Initialize the instance."""
        super().__init__(60, **kwargs)
