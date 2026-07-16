"""Vectorized implementation of the zdt1 problem."""

from .zdt_base import ZDT


class ZDT1(ZDT):
    """Define the ZDT1 benchmark with batched tensor evaluation."""

    name = "ZDT1"

    def __init__(self, **kwargs):
        """Initialize the suite problem and apply its fixed configuration."""
        super().__init__(1, **kwargs)
