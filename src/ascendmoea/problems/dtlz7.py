"""Vectorized implementation of the dtlz7 problem."""

from .dtlz_base import DTLZ


class DTLZ7(DTLZ):
    """Implement the DTLZ7 vectorized optimization problem."""

    name = "DTLZ7"

    def __init__(self, **kwargs):
        """Initialize the instance."""
        super().__init__(7, **kwargs)
