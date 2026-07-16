"""Vectorized implementation of the lsmop2 problem."""

from .lsmop_base import LSMOP


class LSMOP2(LSMOP):
    """Define the LSMOP2 benchmark with batched tensor evaluation."""

    name = "LSMOP2"

    def __init__(self, **kwargs):
        """Initialize the suite problem and apply its fixed configuration."""
        super().__init__(2, **kwargs)
