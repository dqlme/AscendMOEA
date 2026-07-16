"""Vectorized implementation of the mmmop6a problem."""

from __future__ import annotations

from .mmo_base import MMOProblem


class MMMOP6A(MMOProblem):
    """Define the MMMOP6A benchmark with batched tensor evaluation."""

    name = "MMMOP6A"

    def __init__(self, **kwargs):
        """Initialize the suite problem and apply its fixed configuration."""
        super().__init__(59, **kwargs)
