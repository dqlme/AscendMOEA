"""Vectorized implementation of the lsmop2 problem."""

from .lsmop_base import LSMOP


class LSMOP2(LSMOP):
    """Implement the LSMOP2 vectorized optimization problem."""

    name = "LSMOP2"

    def __init__(self, **kwargs):
        """Initialize the instance."""
        super().__init__(2, **kwargs)
