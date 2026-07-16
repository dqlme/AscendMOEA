"""Vectorized implementation of the mmmop4d problem."""

from __future__ import annotations

from .mmo_base import MMOProblem


class MMMOP4D(MMOProblem):
    """Define the MMMOP4D benchmark with batched tensor evaluation."""

    name = "MMMOP4D"

    def __init__(self, **kwargs):
        """Initialize the suite problem and apply its fixed configuration."""
        super().__init__(54, **kwargs)
