"""Vectorized implementation of the mmf10 problem."""

from __future__ import annotations

from .mmo_base import MMOProblem


class MMF10(MMOProblem):
    """Define the MMF10 benchmark with batched tensor evaluation."""

    name = "MMF10"

    def __init__(self, **kwargs):
        """Initialize the suite problem and apply its fixed configuration."""
        super().__init__(12, **kwargs)
