"""Vectorized implementation of the dtlz6 problem."""

from .dtlz_base import DTLZ


class DTLZ6(DTLZ):
    """Implement the DTLZ6 vectorized optimization problem."""

    name = "DTLZ6"

    def __init__(self, **kwargs):
        """Initialize the instance."""
        super().__init__(6, **kwargs)
