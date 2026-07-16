"""Vectorized implementation of the lsmop3 problem."""

from .lsmop_base import LSMOP


class LSMOP3(LSMOP):
    """Implement the LSMOP3 vectorized optimization problem."""

    name = "LSMOP3"

    def __init__(self, **kwargs):
        """Initialize the instance."""
        super().__init__(3, **kwargs)
