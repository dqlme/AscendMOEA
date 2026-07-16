"""Vectorized implementation of the mmmop6b problem."""

from __future__ import annotations

from .mmo_base import MMOProblem


class MMMOP6B(MMOProblem):
    """Define the MMMOP6B benchmark with batched tensor evaluation."""

    name = "MMMOP6B"

    def __init__(self, **kwargs):
        """Initialize the suite problem and apply its fixed configuration."""
        super().__init__(60, **kwargs)
