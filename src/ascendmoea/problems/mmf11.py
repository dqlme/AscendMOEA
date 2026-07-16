"""Vectorized implementation of the mmf11 problem."""

from __future__ import annotations

from .mmo_base import MMOProblem


class MMF11(MMOProblem):
    """Define the MMF11 benchmark with batched tensor evaluation."""

    name = "MMF11"

    def __init__(self, **kwargs):
        """Initialize the suite problem and apply its fixed configuration."""
        super().__init__(13, **kwargs)
