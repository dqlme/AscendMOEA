"""Vectorized implementation of the idmpm2t2 problem."""

from __future__ import annotations

from .mmo_base import MMOProblem


class IDMPM2T2(MMOProblem):
    """Define the IDMPM2T2 benchmark with batched tensor evaluation."""

    name = "IDMPM2T2"

    def __init__(self, **kwargs):
        """Initialize the suite problem and apply its fixed configuration."""
        super().__init__(24, **kwargs)
