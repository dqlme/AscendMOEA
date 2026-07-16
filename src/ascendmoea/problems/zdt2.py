"""Vectorized implementation of the zdt2 problem."""

from .zdt_base import ZDT


class ZDT2(ZDT):
    """Implement the ZDT2 vectorized optimization problem."""

    name = "ZDT2"

    def __init__(self, **kwargs):
        """Initialize the instance."""
        super().__init__(2, **kwargs)
