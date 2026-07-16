"""Vectorized implementation of the mmmop4d problem."""

from __future__ import annotations

from .mmo_base import MMOProblem


class MMMOP4D(MMOProblem):
    """Implement the MMMOP4D vectorized optimization problem."""

    name = "MMMOP4D"

    def __init__(self, **kwargs):
        """Initialize the instance."""
        super().__init__(54, **kwargs)
