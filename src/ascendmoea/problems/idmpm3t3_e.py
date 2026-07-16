"""Vectorized implementation of the idmpm3t3_e problem."""

from __future__ import annotations

from .mmo_base import MMOProblem


class IDMPM3T3E(MMOProblem):
    """Define the IDMPM3T3E benchmark with batched tensor evaluation."""

    name = "IDMPM3T3_e"

    def __init__(self, **kwargs):
        """Initialize the suite problem and apply its fixed configuration."""
        super().__init__(41, **kwargs)
