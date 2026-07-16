"""Vectorized implementation of the mmf1_e problem."""

from __future__ import annotations

from .mmo_base import MMOProblem


class MMF1E(MMOProblem):
    """Define the MMF1E benchmark with batched tensor evaluation."""

    name = "MMF1_e"

    def __init__(self, **kwargs):
        """Initialize the suite problem and apply its fixed configuration."""
        super().__init__(2, **kwargs)
