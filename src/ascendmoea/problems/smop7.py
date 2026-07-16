"""Vectorized implementation of the smop7 problem."""

from .smop_base import SMOP


class SMOP7(SMOP):
    """Implement the SMOP7 vectorized optimization problem."""

    name = "SMOP7"

    def __init__(self, **kwargs):
        """Initialize the instance."""
        super().__init__(7, **kwargs)
