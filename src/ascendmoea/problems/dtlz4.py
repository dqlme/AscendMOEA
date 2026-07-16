"""Vectorized implementation of the dtlz4 problem."""

from .dtlz_base import DTLZ


class DTLZ4(DTLZ):
    """Define the DTLZ4 benchmark with batched tensor evaluation."""

    name = "DTLZ4"

    def __init__(self, **kwargs):
        """Initialize the suite problem and apply its fixed configuration."""
        super().__init__(4, **kwargs)
