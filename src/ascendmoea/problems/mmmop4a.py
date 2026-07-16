"""Vectorized implementation of the mmmop4a problem."""

from __future__ import annotations

from .mmo_base import MMOProblem


class MMMOP4A(MMOProblem):
    """Implement the MMMOP4A vectorized optimization problem."""

    name = "MMMOP4A"

    def __init__(self, **kwargs):
        """Initialize the instance."""
        super().__init__(51, **kwargs)
