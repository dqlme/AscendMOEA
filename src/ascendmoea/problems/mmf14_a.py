"""Vectorized implementation of the mmf14_a problem."""

from __future__ import annotations

from .mmo_base import MMOProblem


class MMF14A(MMOProblem):
    """Define the MMF14A benchmark with batched tensor evaluation."""

    name = "MMF14_a"

    def __init__(self, **kwargs):
        """Initialize the suite problem and apply its fixed configuration."""
        super().__init__(17, **kwargs)
