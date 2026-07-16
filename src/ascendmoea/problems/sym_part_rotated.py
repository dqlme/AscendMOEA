"""Vectorized implementation of the sym_part_rotated problem."""

from __future__ import annotations

from .mmo_base import MMOProblem


class SYMPARTRotated(MMOProblem):
    """Implement the SYMPARTRotated vectorized optimization problem."""

    name = "SYM_PART_rotated"

    def __init__(self, **kwargs):
        """Initialize the instance."""
        super().__init__(21, **kwargs)
