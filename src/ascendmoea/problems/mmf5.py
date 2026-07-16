"""Vectorized implementation of the mmf5 problem."""

from __future__ import annotations

from .mmo_base import MMOProblem


class MMF5(MMOProblem):
    """Define the MMF5 benchmark with batched tensor evaluation."""

    name = "MMF5"

    def __init__(self, **kwargs):
        """Initialize the suite problem and apply its fixed configuration."""
        super().__init__(7, **kwargs)
