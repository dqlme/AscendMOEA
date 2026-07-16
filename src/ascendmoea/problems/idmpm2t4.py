"""Vectorized implementation of the idmpm2t4 problem."""

from __future__ import annotations

from .mmo_base import MMOProblem


class IDMPM2T4(MMOProblem):
    """Implement the IDMPM2T4 vectorized optimization problem."""

    name = "IDMPM2T4"

    def __init__(self, **kwargs):
        """Initialize the instance."""
        super().__init__(26, **kwargs)
