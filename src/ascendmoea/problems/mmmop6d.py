"""Vectorized implementation of the mmmop6d problem."""

from __future__ import annotations

from .mmo_base import MMOProblem


class MMMOP6D(MMOProblem):
    """Define the MMMOP6D benchmark with batched tensor evaluation."""

    name = "MMMOP6D"

    def __init__(self, **kwargs):
        """Initialize the suite problem and apply its fixed configuration."""
        super().__init__(62, **kwargs)
