"""Vectorized implementation of the mmmop2b problem."""

from __future__ import annotations

from .mmo_base import MMOProblem


class MMMOP2B(MMOProblem):
    """Define the MMMOP2B benchmark with batched tensor evaluation."""

    name = "MMMOP2B"

    def __init__(self, **kwargs):
        """Initialize the suite problem and apply its fixed configuration."""
        super().__init__(46, **kwargs)
