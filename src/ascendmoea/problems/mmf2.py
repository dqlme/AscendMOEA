"""Vectorized implementation of the mmf2 problem."""

from __future__ import annotations

from .mmo_base import MMOProblem


class MMF2(MMOProblem):
    """Define the MMF2 benchmark with batched tensor evaluation."""

    name = "MMF2"

    def __init__(self, **kwargs):
        """Initialize the suite problem and apply its fixed configuration."""
        super().__init__(4, **kwargs)
