"""Vectorized implementation of the zdt4 problem."""

from .zdt_base import ZDT


class ZDT4(ZDT):
    """Define the ZDT4 benchmark with batched tensor evaluation."""

    name = "ZDT4"

    def __init__(self, **kwargs):
        """Initialize the suite problem and apply its fixed configuration."""
        super().__init__(4, **kwargs)
