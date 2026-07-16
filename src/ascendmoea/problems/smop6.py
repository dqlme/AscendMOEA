"""Vectorized implementation of the smop6 problem."""

from .smop_base import SMOP


class SMOP6(SMOP):
    """Define the SMOP6 benchmark with batched tensor evaluation."""

    name = "SMOP6"

    def __init__(self, **kwargs):
        """Initialize the suite problem and apply its fixed configuration."""
        super().__init__(6, **kwargs)
