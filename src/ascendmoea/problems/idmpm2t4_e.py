"""Vectorized implementation of the idmpm2t4_e problem."""

from __future__ import annotations

from .mmo_base import MMOProblem


class IDMPM2T4E(MMOProblem):
    """Implement the IDMPM2T4E vectorized optimization problem."""

    name = "IDMPM2T4_e"

    def __init__(self, **kwargs):
        """Initialize the instance."""
        super().__init__(38, **kwargs)
