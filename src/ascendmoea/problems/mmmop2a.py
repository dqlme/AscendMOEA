"""Vectorized implementation of the mmmop2a problem."""

from __future__ import annotations

from .mmo_base import MMOProblem


class MMMOP2A(MMOProblem):
    """Define the MMMOP2A benchmark with batched tensor evaluation."""

    name = "MMMOP2A"

    def __init__(self, **kwargs):
        """Initialize the suite problem and apply its fixed configuration."""
        super().__init__(45, **kwargs)
