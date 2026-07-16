"""Vectorized implementation of the mmmop3b problem."""

from __future__ import annotations

from .mmo_base import MMOProblem


class MMMOP3B(MMOProblem):
    """Define the MMMOP3B benchmark with batched tensor evaluation."""

    name = "MMMOP3B"

    def __init__(self, **kwargs):
        """Initialize the suite problem and apply its fixed configuration."""
        super().__init__(48, **kwargs)
