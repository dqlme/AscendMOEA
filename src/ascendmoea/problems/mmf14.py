"""Vectorized implementation of the mmf14 problem."""

from __future__ import annotations

from .mmo_base import MMOProblem


class MMF14(MMOProblem):
    """Define the MMF14 benchmark with batched tensor evaluation."""

    name = "MMF14"

    def __init__(self, **kwargs):
        """Initialize the suite problem and apply its fixed configuration."""
        super().__init__(16, **kwargs)
