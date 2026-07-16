"""Vectorized implementation of the mmmop5d problem."""

from __future__ import annotations

from .mmo_base import MMOProblem


class MMMOP5D(MMOProblem):
    """Define the MMMOP5D benchmark with batched tensor evaluation."""

    name = "MMMOP5D"

    def __init__(self, **kwargs):
        """Initialize the suite problem and apply its fixed configuration."""
        super().__init__(58, **kwargs)
