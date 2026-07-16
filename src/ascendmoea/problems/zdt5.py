"""Vectorized implementation of the zdt5 problem."""

from .zdt_base import ZDT


class ZDT5(ZDT):
    """Define the ZDT5 benchmark with batched tensor evaluation."""

    name = "ZDT5"

    def __init__(self, **kwargs):
        """Initialize the suite problem and apply its fixed configuration."""
        super().__init__(5, **kwargs)
