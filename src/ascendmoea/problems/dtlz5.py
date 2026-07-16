"""Vectorized implementation of the dtlz5 problem."""

from .dtlz_base import DTLZ


class DTLZ5(DTLZ):
    """Implement the DTLZ5 vectorized optimization problem."""

    name = "DTLZ5"

    def __init__(self, **kwargs):
        """Initialize the instance."""
        super().__init__(5, **kwargs)
