"""Vectorized implementation of the idmpm4t4 problem."""

from __future__ import annotations

from .mmo_base import MMOProblem


class IDMPM4T4(MMOProblem):
    """Define the IDMPM4T4 benchmark with batched tensor evaluation."""

    name = "IDMPM4T4"

    def __init__(self, **kwargs):
        """Initialize the suite problem and apply its fixed configuration."""
        super().__init__(34, **kwargs)
