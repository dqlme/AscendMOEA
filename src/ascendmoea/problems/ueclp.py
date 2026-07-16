"""Vectorized implementation of the ueclp problem."""

from __future__ import annotations

from .mmo_base import MMOProblem


class UECLP(MMOProblem):
    """Define the UECLP benchmark with batched tensor evaluation."""

    name = "UECLP"

    def __init__(self, **kwargs):
        """Initialize the suite problem and apply its fixed configuration."""
        super().__init__(74, **kwargs)
