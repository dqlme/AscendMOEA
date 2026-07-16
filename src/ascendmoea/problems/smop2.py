"""Vectorized implementation of the smop2 problem."""

from __future__ import annotations

from .smop_base import SMOP


class SMOP2(SMOP):
    """Define the SMOP2 benchmark with batched tensor evaluation."""

    name = "SMOP2"

    def __init__(self, **kwargs):
        """Initialize the suite problem and apply its fixed configuration."""
        super().__init__(2, **kwargs)
