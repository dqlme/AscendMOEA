"""Vectorized implementation of the mmmop6c problem."""

from __future__ import annotations

from .mmo_base import MMOProblem


class MMMOP6C(MMOProblem):
    """Implement the MMMOP6C vectorized optimization problem."""

    name = "MMMOP6C"

    def __init__(self, **kwargs):
        """Initialize the instance."""
        super().__init__(61, **kwargs)
