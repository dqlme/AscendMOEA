"""Vectorized implementation of the zdt3 problem."""

from .zdt_base import ZDT


class ZDT3(ZDT):
    """Implement the ZDT3 vectorized optimization problem."""

    name = "ZDT3"

    def __init__(self, **kwargs):
        """Initialize the instance."""
        super().__init__(3, **kwargs)
