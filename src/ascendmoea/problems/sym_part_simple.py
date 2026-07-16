"""Vectorized implementation of the sym_part_simple problem."""

from __future__ import annotations

from .mmo_base import MMOProblem


class SYMPARTSimple(MMOProblem):
    """Implement the SYMPARTSimple vectorized optimization problem."""

    name = "SYM_PART_simple"

    def __init__(self, **kwargs):
        """Initialize the instance."""
        super().__init__(20, **kwargs)
