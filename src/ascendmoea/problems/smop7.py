"""Vectorized implementation of the smop7 problem."""

from .smop_base import SMOP


class SMOP7(SMOP):
    """Define the SMOP7 benchmark with batched tensor evaluation."""

    name = "SMOP7"

    def __init__(self, **kwargs):
        """Initialize the suite problem and apply its fixed configuration."""
        super().__init__(7, **kwargs)
