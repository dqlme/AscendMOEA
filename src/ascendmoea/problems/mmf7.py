"""Vectorized implementation of the mmf7 problem."""

from __future__ import annotations

from .mmo_base import MMOProblem


class MMF7(MMOProblem):
    """Define the MMF7 benchmark with batched tensor evaluation."""

    name = "MMF7"

    def __init__(self, **kwargs):
        """Initialize the suite problem and apply its fixed configuration."""
        super().__init__(9, **kwargs)
