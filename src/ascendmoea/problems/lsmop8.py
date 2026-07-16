"""Vectorized implementation of the lsmop8 problem."""

from .lsmop_base import LSMOP


class LSMOP8(LSMOP):
    """Implement the LSMOP8 vectorized optimization problem."""

    name = "LSMOP8"

    def __init__(self, **kwargs):
        """Initialize the instance."""
        super().__init__(8, **kwargs)
