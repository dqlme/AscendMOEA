"""Vectorized implementation of the mmmop2a problem."""

from __future__ import annotations

from .mmo_base import MMOProblem


class MMMOP2A(MMOProblem):
    """Implement the MMMOP2A vectorized optimization problem."""

    name = "MMMOP2A"

    def __init__(self, **kwargs):
        """Initialize the instance."""
        super().__init__(45, **kwargs)
