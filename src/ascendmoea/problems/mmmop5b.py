"""Vectorized implementation of the mmmop5b problem."""

from __future__ import annotations

from .mmo_base import MMOProblem


class MMMOP5B(MMOProblem):
    """Define the MMMOP5B benchmark with batched tensor evaluation."""

    name = "MMMOP5B"

    def __init__(self, **kwargs):
        """Initialize the suite problem and apply its fixed configuration."""
        super().__init__(56, **kwargs)
