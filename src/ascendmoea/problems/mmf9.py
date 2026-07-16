"""Vectorized implementation of the mmf9 problem."""

from __future__ import annotations

from .mmo_base import MMOProblem


class MMF9(MMOProblem):
    """Define the MMF9 benchmark with batched tensor evaluation."""

    name = "MMF9"

    def __init__(self, **kwargs):
        """Initialize the suite problem and apply its fixed configuration."""
        super().__init__(11, **kwargs)
