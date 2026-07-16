"""Vectorized implementation of the smop8 problem."""

from .smop_base import SMOP


class SMOP8(SMOP):
    """Define the SMOP8 benchmark with batched tensor evaluation."""

    name = "SMOP8"

    def __init__(self, **kwargs):
        """Initialize the suite problem and apply its fixed configuration."""
        super().__init__(8, **kwargs)
