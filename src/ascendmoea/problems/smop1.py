"""Vectorized implementation of the smop1 problem."""

from __future__ import annotations

from .smop_base import SMOP


class SMOP1(SMOP):
    """Implement the SMOP1 vectorized optimization problem."""

    name = "SMOP1"

    def __init__(self, **kwargs):
        """Initialize the instance."""
        super().__init__(1, **kwargs)
