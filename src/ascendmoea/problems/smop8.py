"""Vectorized implementation of the smop8 problem."""

from .smop_base import SMOP


class SMOP8(SMOP):
    """Implement the SMOP8 vectorized optimization problem."""

    name = "SMOP8"

    def __init__(self, **kwargs):
        """Initialize the instance."""
        super().__init__(8, **kwargs)
