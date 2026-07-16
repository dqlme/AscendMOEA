"""Vectorized implementation of the smop5 problem."""

from .smop_base import SMOP


class SMOP5(SMOP):
    """Implement the SMOP5 vectorized optimization problem."""

    name = "SMOP5"

    def __init__(self, **kwargs):
        """Initialize the instance."""
        super().__init__(5, **kwargs)
