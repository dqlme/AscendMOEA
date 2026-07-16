"""Vectorized implementation of the mmf15_a problem."""

from __future__ import annotations

from .mmo_base import MMOProblem


class MMF15A(MMOProblem):
    """Define the MMF15A benchmark with batched tensor evaluation."""

    name = "MMF15_a"

    def __init__(self, **kwargs):
        """Initialize the suite problem and apply its fixed configuration."""
        super().__init__(19, **kwargs)
