"""Vectorized implementation of the mmmop3a problem."""

from __future__ import annotations

from .mmo_base import MMOProblem


class MMMOP3A(MMOProblem):
    """Implement the MMMOP3A vectorized optimization problem."""

    name = "MMMOP3A"

    def __init__(self, **kwargs):
        """Initialize the instance."""
        super().__init__(47, **kwargs)
