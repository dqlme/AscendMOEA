"""Vectorized implementation of the lsmop7 problem."""

from .lsmop_base import LSMOP


class LSMOP7(LSMOP):
    """Implement the LSMOP7 vectorized optimization problem."""

    name = "LSMOP7"

    def __init__(self, **kwargs):
        """Initialize the instance."""
        super().__init__(7, **kwargs)
