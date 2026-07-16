"""Vectorized implementation of the mmmop1a problem."""

from __future__ import annotations

from .mmo_base import MMOProblem


class MMMOP1A(MMOProblem):
    """Implement the MMMOP1A vectorized optimization problem."""

    name = "MMMOP1A"

    def __init__(self, **kwargs):
        """Initialize the instance."""
        super().__init__(43, **kwargs)
