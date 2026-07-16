"""Vectorized implementation of the idmpm3t2 problem."""

from __future__ import annotations

from .mmo_base import MMOProblem


class IDMPM3T2(MMOProblem):
    """Define the IDMPM3T2 benchmark with batched tensor evaluation."""

    name = "IDMPM3T2"

    def __init__(self, **kwargs):
        """Initialize the suite problem and apply its fixed configuration."""
        super().__init__(28, **kwargs)
