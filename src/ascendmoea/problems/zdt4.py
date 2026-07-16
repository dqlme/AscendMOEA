"""Vectorized implementation of the zdt4 problem."""

from .zdt_base import ZDT


class ZDT4(ZDT):
    """Implement the ZDT4 vectorized optimization problem."""

    name = "ZDT4"

    def __init__(self, **kwargs):
        """Initialize the instance."""
        super().__init__(4, **kwargs)
