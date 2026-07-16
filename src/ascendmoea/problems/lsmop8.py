"""Vectorized implementation of the lsmop8 problem."""

from .lsmop_base import LSMOP


class LSMOP8(LSMOP):
    """Define the LSMOP8 benchmark with batched tensor evaluation."""

    name = "LSMOP8"

    def __init__(self, **kwargs):
        """Initialize the suite problem and apply its fixed configuration."""
        super().__init__(8, **kwargs)
