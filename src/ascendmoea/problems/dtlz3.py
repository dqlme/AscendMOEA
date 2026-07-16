"""Vectorized implementation of the dtlz3 problem."""

from .dtlz_base import DTLZ


class DTLZ3(DTLZ):
    """Implement the DTLZ3 vectorized optimization problem."""

    name = "DTLZ3"

    def __init__(self, **kwargs):
        """Initialize the instance."""
        super().__init__(3, **kwargs)
