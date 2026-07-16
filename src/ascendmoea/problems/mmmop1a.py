"""Vectorized implementation of the mmmop1a problem."""

from __future__ import annotations

from .mmo_base import MMOProblem


class MMMOP1A(MMOProblem):
    """Define the MMMOP1A benchmark with batched tensor evaluation."""

    name = "MMMOP1A"

    def __init__(self, **kwargs):
        """Initialize the suite problem and apply its fixed configuration."""
        super().__init__(43, **kwargs)
