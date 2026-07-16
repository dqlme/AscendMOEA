"""Vectorized implementation of the lsmop1 problem."""

from .lsmop_base import LSMOP


class LSMOP1(LSMOP):
    """Implement the LSMOP1 vectorized optimization problem."""

    name = "LSMOP1"

    def __init__(self, **kwargs):
        """Initialize the instance."""
        super().__init__(1, **kwargs)
