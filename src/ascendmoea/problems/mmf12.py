"""Vectorized implementation of the mmf12 problem."""

from __future__ import annotations

from .mmo_base import MMOProblem


class MMF12(MMOProblem):
    """Define the MMF12 benchmark with batched tensor evaluation."""

    name = "MMF12"

    def __init__(self, **kwargs):
        """Initialize the suite problem and apply its fixed configuration."""
        super().__init__(14, **kwargs)
