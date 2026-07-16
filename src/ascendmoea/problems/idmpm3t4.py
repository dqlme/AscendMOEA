"""Vectorized implementation of the idmpm3t4 problem."""

from __future__ import annotations

from .mmo_base import MMOProblem


class IDMPM3T4(MMOProblem):
    """Implement the IDMPM3T4 vectorized optimization problem."""

    name = "IDMPM3T4"

    def __init__(self, **kwargs):
        """Initialize the instance."""
        super().__init__(30, **kwargs)
