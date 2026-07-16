"""Vectorized implementation of the smop1 problem."""

from __future__ import annotations

from .smop_base import SMOP


class SMOP1(SMOP):
    """Define the SMOP1 benchmark with batched tensor evaluation."""

    name = "SMOP1"

    def __init__(self, **kwargs):
        """Initialize the suite problem and apply its fixed configuration."""
        super().__init__(1, **kwargs)
