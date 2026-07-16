"""Vectorized implementation of the mmmop5a problem."""

from __future__ import annotations

from .mmo_base import MMOProblem


class MMMOP5A(MMOProblem):
    """Define the MMMOP5A benchmark with batched tensor evaluation."""

    name = "MMMOP5A"

    def __init__(self, **kwargs):
        """Initialize the suite problem and apply its fixed configuration."""
        super().__init__(55, **kwargs)
