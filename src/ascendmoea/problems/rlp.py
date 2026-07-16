"""Vectorized implementation of the rlp problem."""

from __future__ import annotations

from .mmo_base import MMOProblem


class RLP(MMOProblem):
    """Implement the RLP vectorized optimization problem."""

    name = "RLP"

    def __init__(self, **kwargs):
        """Initialize the instance."""
        super().__init__(73, **kwargs)
