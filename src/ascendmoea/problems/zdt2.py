"""Vectorized implementation of the zdt2 problem."""

from .zdt_base import ZDT


class ZDT2(ZDT):
    """Define the ZDT2 benchmark with batched tensor evaluation."""

    name = "ZDT2"

    def __init__(self, **kwargs):
        """Initialize the suite problem and apply its fixed configuration."""
        super().__init__(2, **kwargs)
