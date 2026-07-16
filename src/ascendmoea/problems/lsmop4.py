"""Vectorized implementation of the lsmop4 problem."""

from .lsmop_base import LSMOP


class LSMOP4(LSMOP):
    """Define the LSMOP4 benchmark with batched tensor evaluation."""

    name = "LSMOP4"

    def __init__(self, **kwargs):
        """Initialize the suite problem and apply its fixed configuration."""
        super().__init__(4, **kwargs)
