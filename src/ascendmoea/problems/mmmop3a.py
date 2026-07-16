"""Vectorized implementation of the mmmop3a problem."""

from __future__ import annotations

from .mmo_base import MMOProblem


class MMMOP3A(MMOProblem):
    """Define the MMMOP3A benchmark with batched tensor evaluation."""

    name = "MMMOP3A"

    def __init__(self, **kwargs):
        """Initialize the suite problem and apply its fixed configuration."""
        super().__init__(47, **kwargs)
