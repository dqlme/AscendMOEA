"""Vectorized implementation of the dtlz3 problem."""

from .dtlz_base import DTLZ


class DTLZ3(DTLZ):
    """Define the DTLZ3 benchmark with batched tensor evaluation."""

    name = "DTLZ3"

    def __init__(self, **kwargs):
        """Initialize the suite problem and apply its fixed configuration."""
        super().__init__(3, **kwargs)
