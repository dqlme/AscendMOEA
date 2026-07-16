"""Vectorized implementation of the mmmop4c problem."""

from __future__ import annotations

from .mmo_base import MMOProblem


class MMMOP4C(MMOProblem):
    """Define the MMMOP4C benchmark with batched tensor evaluation."""

    name = "MMMOP4C"

    def __init__(self, **kwargs):
        """Initialize the suite problem and apply its fixed configuration."""
        super().__init__(53, **kwargs)
