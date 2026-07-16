"""Vectorized implementation of the idmpm4t3 problem."""

from __future__ import annotations

from .mmo_base import MMOProblem


class IDMPM4T3(MMOProblem):
    """Define the IDMPM4T3 benchmark with batched tensor evaluation."""

    name = "IDMPM4T3"

    def __init__(self, **kwargs):
        """Initialize the suite problem and apply its fixed configuration."""
        super().__init__(33, **kwargs)
