"""Vectorized implementation of the dtlz4 problem."""

from .dtlz_base import DTLZ


class DTLZ4(DTLZ):
    """Implement the DTLZ4 vectorized optimization problem."""

    name = "DTLZ4"

    def __init__(self, **kwargs):
        """Initialize the instance."""
        super().__init__(4, **kwargs)
