"""Vectorized implementation of the lsmop9 problem."""

from .lsmop_base import LSMOP


class LSMOP9(LSMOP):
    """Implement the LSMOP9 vectorized optimization problem."""

    name = "LSMOP9"

    def __init__(self, **kwargs):
        """Initialize the instance."""
        super().__init__(9, **kwargs)
