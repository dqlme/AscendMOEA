"""Vectorized implementation of the idmpm2t1 problem."""

from __future__ import annotations

from .mmo_base import MMOProblem


class IDMPM2T1(MMOProblem):
    """Implement the IDMPM2T1 vectorized optimization problem."""

    name = "IDMPM2T1"

    def __init__(self, **kwargs):
        """Initialize the instance."""
        super().__init__(23, **kwargs)
