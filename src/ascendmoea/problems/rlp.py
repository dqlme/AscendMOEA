"""Vectorized implementation of the rlp problem."""

from __future__ import annotations

from .mmo_base import MMOProblem


class RLP(MMOProblem):
    """Define the RLP benchmark with batched tensor evaluation."""

    name = "RLP"

    def __init__(self, **kwargs):
        """Initialize the suite problem and apply its fixed configuration."""
        super().__init__(73, **kwargs)
