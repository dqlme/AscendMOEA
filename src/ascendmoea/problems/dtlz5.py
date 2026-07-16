"""Vectorized implementation of the dtlz5 problem."""

from .dtlz_base import DTLZ


class DTLZ5(DTLZ):
    """Define the DTLZ5 benchmark with batched tensor evaluation."""

    name = "DTLZ5"

    def __init__(self, **kwargs):
        """Initialize the suite problem and apply its fixed configuration."""
        super().__init__(5, **kwargs)
