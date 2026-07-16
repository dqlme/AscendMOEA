"""Vectorized implementation of the smop2 problem."""

from __future__ import annotations

from .smop_base import SMOP


class SMOP2(SMOP):
    """Implement the SMOP2 vectorized optimization problem."""

    name = "SMOP2"

    def __init__(self, **kwargs):
        """Initialize the instance."""
        super().__init__(2, **kwargs)
