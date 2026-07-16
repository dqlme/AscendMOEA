"""Vectorized implementation of the lsmop4 problem."""

from .lsmop_base import LSMOP


class LSMOP4(LSMOP):
    """Implement the LSMOP4 vectorized optimization problem."""

    name = "LSMOP4"

    def __init__(self, **kwargs):
        """Initialize the instance."""
        super().__init__(4, **kwargs)
