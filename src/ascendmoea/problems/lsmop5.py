"""Vectorized implementation of the lsmop5 problem."""

from .lsmop_base import LSMOP


class LSMOP5(LSMOP):
    """Define the LSMOP5 benchmark with batched tensor evaluation."""

    name = "LSMOP5"

    def __init__(self, **kwargs):
        """Initialize the suite problem and apply its fixed configuration."""
        super().__init__(5, **kwargs)
