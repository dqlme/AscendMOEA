"""Vectorized implementation of the mmmop5c problem."""

from __future__ import annotations

from .mmo_base import MMOProblem


class MMMOP5C(MMOProblem):
    """Define the MMMOP5C benchmark with batched tensor evaluation."""

    name = "MMMOP5C"

    def __init__(self, **kwargs):
        """Initialize the suite problem and apply its fixed configuration."""
        super().__init__(57, **kwargs)
