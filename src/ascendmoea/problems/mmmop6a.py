"""Vectorized implementation of the mmmop6a problem."""

from __future__ import annotations

from .mmo_base import MMOProblem


class MMMOP6A(MMOProblem):
    """Implement the MMMOP6A vectorized optimization problem."""

    name = "MMMOP6A"

    def __init__(self, **kwargs):
        """Initialize the instance."""
        super().__init__(59, **kwargs)
