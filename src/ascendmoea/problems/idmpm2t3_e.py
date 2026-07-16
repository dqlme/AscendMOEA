"""Vectorized implementation of the idmpm2t3_e problem."""

from __future__ import annotations

from .mmo_base import MMOProblem


class IDMPM2T3E(MMOProblem):
    """Define the IDMPM2T3E benchmark with batched tensor evaluation."""

    name = "IDMPM2T3_e"

    def __init__(self, **kwargs):
        """Initialize the suite problem and apply its fixed configuration."""
        super().__init__(37, **kwargs)
