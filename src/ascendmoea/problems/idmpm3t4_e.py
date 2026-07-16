"""Vectorized implementation of the idmpm3t4_e problem."""

from __future__ import annotations

from .mmo_base import MMOProblem


class IDMPM3T4E(MMOProblem):
    """Implement the IDMPM3T4E vectorized optimization problem."""

    name = "IDMPM3T4_e"

    def __init__(self, **kwargs):
        """Initialize the instance."""
        super().__init__(42, **kwargs)
