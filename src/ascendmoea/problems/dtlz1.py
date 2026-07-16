"""Vectorized implementation of the dtlz1 problem."""

from .dtlz_base import DTLZ


class DTLZ1(DTLZ):
    """Define the DTLZ1 benchmark with batched tensor evaluation."""

    name = "DTLZ1"

    def __init__(self, **kwargs):
        """Initialize the suite problem and apply its fixed configuration."""
        super().__init__(1, **kwargs)
