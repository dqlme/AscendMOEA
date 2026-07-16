"""Vectorized implementation of the dtlz2 problem."""

from .dtlz_base import DTLZ


class DTLZ2(DTLZ):
    """Define the DTLZ2 benchmark with batched tensor evaluation."""

    name = "DTLZ2"

    def __init__(self, **kwargs):
        """Initialize the suite problem and apply its fixed configuration."""
        super().__init__(2, **kwargs)
