"""Vectorized implementation of the idmpm2t3 problem."""

from __future__ import annotations

from .mmo_base import MMOProblem


class IDMPM2T3(MMOProblem):
    """Define the IDMPM2T3 benchmark with batched tensor evaluation."""

    name = "IDMPM2T3"

    def __init__(self, **kwargs):
        """Initialize the suite problem and apply its fixed configuration."""
        super().__init__(25, **kwargs)
