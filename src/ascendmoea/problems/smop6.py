"""Vectorized implementation of the smop6 problem."""

from .smop_base import SMOP


class SMOP6(SMOP):
    """Implement the SMOP6 vectorized optimization problem."""

    name = "SMOP6"

    def __init__(self, **kwargs):
        """Initialize the instance."""
        super().__init__(6, **kwargs)
