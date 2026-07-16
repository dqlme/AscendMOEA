"""Vectorized implementation of the mmf3 problem."""

from __future__ import annotations

from .mmo_base import MMOProblem


class MMF3(MMOProblem):
    """Define the MMF3 benchmark with batched tensor evaluation."""

    name = "MMF3"

    def __init__(self, **kwargs):
        """Initialize the suite problem and apply its fixed configuration."""
        super().__init__(5, **kwargs)
