"""Vectorized implementation of the mmmop1b problem."""

from __future__ import annotations

from .mmo_base import MMOProblem


class MMMOP1B(MMOProblem):
    """Define the MMMOP1B benchmark with batched tensor evaluation."""

    name = "MMMOP1B"

    def __init__(self, **kwargs):
        """Initialize the suite problem and apply its fixed configuration."""
        super().__init__(44, **kwargs)
