"""Vectorized implementation of the smop5 problem."""

from .smop_base import SMOP


class SMOP5(SMOP):
    """Define the SMOP5 benchmark with batched tensor evaluation."""

    name = "SMOP5"

    def __init__(self, **kwargs):
        """Initialize the suite problem and apply its fixed configuration."""
        super().__init__(5, **kwargs)
