"""Vectorized implementation of the smop4 problem."""

from __future__ import annotations

from .smop_base import SMOP


class SMOP4(SMOP):
    """Implement the SMOP4 vectorized optimization problem."""

    name = "SMOP4"

    def __init__(self, **kwargs):
        """Initialize the instance."""
        super().__init__(4, **kwargs)
