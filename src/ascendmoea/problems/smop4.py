"""Vectorized implementation of the smop4 problem."""

from __future__ import annotations

from .smop_base import SMOP


class SMOP4(SMOP):
    """Define the SMOP4 benchmark with batched tensor evaluation."""

    name = "SMOP4"

    def __init__(self, **kwargs):
        """Initialize the suite problem and apply its fixed configuration."""
        super().__init__(4, **kwargs)
