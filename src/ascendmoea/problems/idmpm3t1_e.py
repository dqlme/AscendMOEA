"""Vectorized implementation of the idmpm3t1_e problem."""

from __future__ import annotations

from .mmo_base import MMOProblem


class IDMPM3T1E(MMOProblem):
    """Implement the IDMPM3T1E vectorized optimization problem."""

    name = "IDMPM3T1_e"

    def __init__(self, **kwargs):
        """Initialize the instance."""
        super().__init__(39, **kwargs)
