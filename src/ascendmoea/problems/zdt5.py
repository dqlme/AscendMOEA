"""Vectorized implementation of the zdt5 problem."""

from .zdt_base import ZDT


class ZDT5(ZDT):
    """Implement the ZDT5 vectorized optimization problem."""

    name = "ZDT5"

    def __init__(self, **kwargs):
        """Initialize the instance."""
        super().__init__(5, **kwargs)
