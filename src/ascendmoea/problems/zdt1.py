"""Vectorized implementation of the zdt1 problem."""

from .zdt_base import ZDT


class ZDT1(ZDT):
    """Implement the ZDT1 vectorized optimization problem."""

    name = "ZDT1"

    def __init__(self, **kwargs):
        """Initialize the instance."""
        super().__init__(1, **kwargs)
