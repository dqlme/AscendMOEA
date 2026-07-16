"""Vectorized implementation of the mmf1 problem."""

from __future__ import annotations

from .mmo_base import MMOProblem


class MMF1(MMOProblem):
    """Define the MMF1 benchmark with batched tensor evaluation."""

    name = "MMF1"

    def __init__(self, **kwargs):
        """Initialize the suite problem and apply its fixed configuration."""
        super().__init__(1, **kwargs)
