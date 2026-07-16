"""Vectorized implementation of the lsmop9 problem."""

from .lsmop_base import LSMOP


class LSMOP9(LSMOP):
    """Define the LSMOP9 benchmark with batched tensor evaluation."""

    name = "LSMOP9"

    def __init__(self, **kwargs):
        """Initialize the suite problem and apply its fixed configuration."""
        super().__init__(9, **kwargs)
