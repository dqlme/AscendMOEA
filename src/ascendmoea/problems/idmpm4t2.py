"""Vectorized implementation of the idmpm4t2 problem."""

from __future__ import annotations

from .mmo_base import MMOProblem


class IDMPM4T2(MMOProblem):
    """Implement the IDMPM4T2 vectorized optimization problem."""

    name = "IDMPM4T2"

    def __init__(self, **kwargs):
        """Initialize the instance."""
        super().__init__(32, **kwargs)
