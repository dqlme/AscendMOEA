"""Vectorized implementation of the zdt3 problem."""

from .zdt_base import ZDT


class ZDT3(ZDT):
    """Define the ZDT3 benchmark with batched tensor evaluation."""

    name = "ZDT3"

    def __init__(self, **kwargs):
        """Initialize the suite problem and apply its fixed configuration."""
        super().__init__(3, **kwargs)
