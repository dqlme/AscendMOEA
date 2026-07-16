"""Vectorized implementation of the mmf8 problem."""

from __future__ import annotations

from .mmo_base import MMOProblem


class MMF8(MMOProblem):
    """Define the MMF8 benchmark with batched tensor evaluation."""

    name = "MMF8"

    def __init__(self, **kwargs):
        """Initialize the suite problem and apply its fixed configuration."""
        super().__init__(10, **kwargs)
