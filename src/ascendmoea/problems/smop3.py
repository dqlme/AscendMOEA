"""Vectorized implementation of the smop3 problem."""

from __future__ import annotations

from .smop_base import SMOP


class SMOP3(SMOP):
    """Define the SMOP3 benchmark with batched tensor evaluation."""

    name = "SMOP3"

    def __init__(self, **kwargs):
        """Initialize the suite problem and apply its fixed configuration."""
        super().__init__(3, **kwargs)
