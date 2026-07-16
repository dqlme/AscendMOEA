"""Vectorized implementation of the lsmop6 problem."""

from .lsmop_base import LSMOP


class LSMOP6(LSMOP):
    """Define the LSMOP6 benchmark with batched tensor evaluation."""

    name = "LSMOP6"

    def __init__(self, **kwargs):
        """Initialize the suite problem and apply its fixed configuration."""
        super().__init__(6, **kwargs)
