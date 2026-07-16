"""Vectorized implementation of the dtlz7 problem."""

from .dtlz_base import DTLZ


class DTLZ7(DTLZ):
    """Define the DTLZ7 benchmark with batched tensor evaluation."""

    name = "DTLZ7"

    def __init__(self, **kwargs):
        """Initialize the suite problem and apply its fixed configuration."""
        super().__init__(7, **kwargs)
