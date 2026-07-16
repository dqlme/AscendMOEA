"""Vectorized implementation of the lsmop3 problem."""

from .lsmop_base import LSMOP


class LSMOP3(LSMOP):
    """Define the LSMOP3 benchmark with batched tensor evaluation."""

    name = "LSMOP3"

    def __init__(self, **kwargs):
        """Initialize the suite problem and apply its fixed configuration."""
        super().__init__(3, **kwargs)
