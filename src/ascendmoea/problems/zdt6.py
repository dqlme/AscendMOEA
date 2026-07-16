"""Vectorized implementation of the zdt6 problem."""

from .zdt_base import ZDT


class ZDT6(ZDT):
    """Implement the ZDT6 vectorized optimization problem."""

    name = "ZDT6"

    def __init__(self, **kwargs):
        """Initialize the instance."""
        super().__init__(6, **kwargs)
