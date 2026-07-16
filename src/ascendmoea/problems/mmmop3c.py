"""Vectorized implementation of the mmmop3c problem."""

from __future__ import annotations

from .mmo_base import MMOProblem


class MMMOP3C(MMOProblem):
    """Define the MMMOP3C benchmark with batched tensor evaluation."""

    name = "MMMOP3C"

    def __init__(self, **kwargs):
        """Initialize the suite problem and apply its fixed configuration."""
        super().__init__(49, **kwargs)
