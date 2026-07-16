"""Vectorized implementation of the idmpm2t2 problem."""

from __future__ import annotations

from .mmo_base import MMOProblem


class IDMPM2T2(MMOProblem):
    """Implement the IDMPM2T2 vectorized optimization problem."""

    name = "IDMPM2T2"

    def __init__(self, **kwargs):
        """Initialize the instance."""
        super().__init__(24, **kwargs)
