"""Vectorized implementation of the dtlz2 problem."""

from .dtlz_base import DTLZ


class DTLZ2(DTLZ):
    """Implement the DTLZ2 vectorized optimization problem."""

    name = "DTLZ2"

    def __init__(self, **kwargs):
        """Initialize the instance."""
        super().__init__(2, **kwargs)
