"""Vectorized implementation of the mmf1_z problem."""

from __future__ import annotations

from .mmo_base import MMOProblem


class MMF1Z(MMOProblem):
    """Define the MMF1Z benchmark with batched tensor evaluation."""

    name = "MMF1_z"

    def __init__(self, **kwargs):
        """Initialize the suite problem and apply its fixed configuration."""
        super().__init__(3, **kwargs)
