"""Vectorized implementation of the lsmop6 problem."""

from .lsmop_base import LSMOP


class LSMOP6(LSMOP):
    """Implement the LSMOP6 vectorized optimization problem."""

    name = "LSMOP6"

    def __init__(self, **kwargs):
        """Initialize the instance."""
        super().__init__(6, **kwargs)
