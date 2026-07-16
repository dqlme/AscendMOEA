"""Vectorized implementation of the lsmop1 problem."""

from .lsmop_base import LSMOP


class LSMOP1(LSMOP):
    """Define the LSMOP1 benchmark with batched tensor evaluation."""

    name = "LSMOP1"

    def __init__(self, **kwargs):
        """Initialize the suite problem and apply its fixed configuration."""
        super().__init__(1, **kwargs)
