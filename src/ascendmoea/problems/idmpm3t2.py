"""Vectorized implementation of the idmpm3t2 problem."""

from __future__ import annotations

from .mmo_base import MMOProblem


class IDMPM3T2(MMOProblem):
    """Implement the IDMPM3T2 vectorized optimization problem."""

    name = "IDMPM3T2"

    def __init__(self, **kwargs):
        """Initialize the instance."""
        super().__init__(28, **kwargs)
