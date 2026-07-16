"""Vectorized implementation of the lsmop5 problem."""

from .lsmop_base import LSMOP


class LSMOP5(LSMOP):
    """Implement the LSMOP5 vectorized optimization problem."""

    name = "LSMOP5"

    def __init__(self, **kwargs):
        """Initialize the instance."""
        super().__init__(5, **kwargs)
