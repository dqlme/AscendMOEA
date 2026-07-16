"""Vectorized implementation of the sym_part_rotated problem."""

from __future__ import annotations

from .mmo_base import MMOProblem


class SYMPARTRotated(MMOProblem):
    """Define the SYMPARTRotated benchmark with batched tensor evaluation."""

    name = "SYM_PART_rotated"

    def __init__(self, **kwargs):
        """Initialize the suite problem and apply its fixed configuration."""
        super().__init__(21, **kwargs)
