"""Vectorized implementation of the mmf15 problem."""

from __future__ import annotations

from .mmo_base import MMOProblem


class MMF15(MMOProblem):
    """Define the MMF15 benchmark with batched tensor evaluation."""

    name = "MMF15"

    def __init__(self, **kwargs):
        """Initialize the suite problem and apply its fixed configuration."""
        super().__init__(18, **kwargs)
