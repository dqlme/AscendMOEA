"""Vectorized implementation of the mmf13 problem."""

from __future__ import annotations

from .mmo_base import MMOProblem


class MMF13(MMOProblem):
    """Define the MMF13 benchmark with batched tensor evaluation."""

    name = "MMF13"

    def __init__(self, **kwargs):
        """Initialize the suite problem and apply its fixed configuration."""
        super().__init__(15, **kwargs)
