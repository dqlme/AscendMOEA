"""Vectorized implementation of the ueclp problem."""

from __future__ import annotations

from .mmo_base import MMOProblem


class UECLP(MMOProblem):
    """Implement the UECLP vectorized optimization problem."""

    name = "UECLP"

    def __init__(self, **kwargs):
        """Initialize the instance."""
        super().__init__(74, **kwargs)
