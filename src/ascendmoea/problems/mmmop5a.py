"""Vectorized implementation of the mmmop5a problem."""

from __future__ import annotations

from .mmo_base import MMOProblem


class MMMOP5A(MMOProblem):
    """Implement the MMMOP5A vectorized optimization problem."""

    name = "MMMOP5A"

    def __init__(self, **kwargs):
        """Initialize the instance."""
        super().__init__(55, **kwargs)
