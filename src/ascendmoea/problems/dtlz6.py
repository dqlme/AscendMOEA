"""Vectorized implementation of the dtlz6 problem."""

from .dtlz_base import DTLZ


class DTLZ6(DTLZ):
    """Define the DTLZ6 benchmark with batched tensor evaluation."""

    name = "DTLZ6"

    def __init__(self, **kwargs):
        """Initialize the suite problem and apply its fixed configuration."""
        super().__init__(6, **kwargs)
