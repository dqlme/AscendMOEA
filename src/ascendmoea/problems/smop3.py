"""Vectorized implementation of the smop3 problem."""

from __future__ import annotations

from .smop_base import SMOP


class SMOP3(SMOP):
    """Implement the SMOP3 vectorized optimization problem."""

    name = "SMOP3"

    def __init__(self, **kwargs):
        """Initialize the instance."""
        super().__init__(3, **kwargs)
