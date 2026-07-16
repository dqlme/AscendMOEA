"""Vectorized implementation of the idmpm2t1_e problem."""

from __future__ import annotations

from .mmo_base import MMOProblem


class IDMPM2T1E(MMOProblem):
    """Implement the IDMPM2T1E vectorized optimization problem."""

    name = "IDMPM2T1_e"

    def __init__(self, **kwargs):
        """Initialize the instance."""
        super().__init__(35, **kwargs)
