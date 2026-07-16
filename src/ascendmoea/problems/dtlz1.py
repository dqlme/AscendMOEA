"""Vectorized implementation of the dtlz1 problem."""

from .dtlz_base import DTLZ


class DTLZ1(DTLZ):
    """Implement the DTLZ1 vectorized optimization problem."""

    name = "DTLZ1"

    def __init__(self, **kwargs):
        """Initialize the instance."""
        super().__init__(1, **kwargs)
