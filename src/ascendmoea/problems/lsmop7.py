"""Vectorized implementation of the lsmop7 problem."""

from .lsmop_base import LSMOP


class LSMOP7(LSMOP):
    """Define the LSMOP7 benchmark with batched tensor evaluation."""

    name = "LSMOP7"

    def __init__(self, **kwargs):
        """Initialize the suite problem and apply its fixed configuration."""
        super().__init__(7, **kwargs)
