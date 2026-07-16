"""Vectorized implementation of the mmmop4a problem."""

from __future__ import annotations

from .mmo_base import MMOProblem


class MMMOP4A(MMOProblem):
    """Define the MMMOP4A benchmark with batched tensor evaluation."""

    name = "MMMOP4A"

    def __init__(self, **kwargs):
        """Initialize the suite problem and apply its fixed configuration."""
        super().__init__(51, **kwargs)
