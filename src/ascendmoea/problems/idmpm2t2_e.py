"""Vectorized implementation of the idmpm2t2_e problem."""

from __future__ import annotations

from .mmo_base import MMOProblem


class IDMPM2T2E(MMOProblem):
    """Implement the IDMPM2T2E vectorized optimization problem."""

    name = "IDMPM2T2_e"

    def __init__(self, **kwargs):
        """Initialize the instance."""
        super().__init__(36, **kwargs)
