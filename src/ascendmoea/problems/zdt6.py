"""Vectorized implementation of the zdt6 problem."""

from .zdt_base import ZDT


class ZDT6(ZDT):
    """Define the ZDT6 benchmark with batched tensor evaluation."""

    name = "ZDT6"

    def __init__(self, **kwargs):
        """Initialize the suite problem and apply its fixed configuration."""
        super().__init__(6, **kwargs)
