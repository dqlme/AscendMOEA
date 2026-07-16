"""Vectorized implementation of the mmf4 problem."""

from __future__ import annotations

from .mmo_base import MMOProblem


class MMF4(MMOProblem):
    """Define the MMF4 benchmark with batched tensor evaluation."""

    name = "MMF4"

    def __init__(self, **kwargs):
        """Initialize the suite problem and apply its fixed configuration."""
        super().__init__(6, **kwargs)
