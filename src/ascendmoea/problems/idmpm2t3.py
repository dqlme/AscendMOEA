"""Vectorized implementation of the idmpm2t3 problem."""

from __future__ import annotations

from .mmo_base import MMOProblem


class IDMPM2T3(MMOProblem):
    """Implement the IDMPM2T3 vectorized optimization problem."""

    name = "IDMPM2T3"

    def __init__(self, **kwargs):
        """Initialize the instance."""
        super().__init__(25, **kwargs)
