"""Vectorized implementation of the mmmop6c problem."""

from __future__ import annotations

from .mmo_base import MMOProblem


class MMMOP6C(MMOProblem):
    """Define the MMMOP6C benchmark with batched tensor evaluation."""

    name = "MMMOP6C"

    def __init__(self, **kwargs):
        """Initialize the suite problem and apply its fixed configuration."""
        super().__init__(61, **kwargs)
