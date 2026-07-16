"""Vectorized implementation of the sym_part_simple problem."""

from __future__ import annotations

from .mmo_base import MMOProblem


class SYMPARTSimple(MMOProblem):
    """Define the SYMPARTSimple benchmark with batched tensor evaluation."""

    name = "SYM_PART_simple"

    def __init__(self, **kwargs):
        """Initialize the suite problem and apply its fixed configuration."""
        super().__init__(20, **kwargs)
