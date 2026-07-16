"""Vectorized implementation of the idmpm3t3 problem."""

from __future__ import annotations

from .mmo_base import MMOProblem


class IDMPM3T3(MMOProblem):
    """Implement the IDMPM3T3 vectorized optimization problem."""

    name = "IDMPM3T3"

    def __init__(self, **kwargs):
        """Initialize the instance."""
        super().__init__(29, **kwargs)
