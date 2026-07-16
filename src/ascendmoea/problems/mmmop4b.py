"""Vectorized implementation of the mmmop4b problem."""

from __future__ import annotations

from .mmo_base import MMOProblem


class MMMOP4B(MMOProblem):
    """Define the MMMOP4B benchmark with batched tensor evaluation."""

    name = "MMMOP4B"

    def __init__(self, **kwargs):
        """Initialize the suite problem and apply its fixed configuration."""
        super().__init__(52, **kwargs)
