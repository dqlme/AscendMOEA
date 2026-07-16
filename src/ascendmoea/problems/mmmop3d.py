"""Vectorized implementation of the mmmop3d problem."""

from __future__ import annotations

from .mmo_base import MMOProblem


class MMMOP3D(MMOProblem):
    """Define the MMMOP3D benchmark with batched tensor evaluation."""

    name = "MMMOP3D"

    def __init__(self, **kwargs):
        """Initialize the suite problem and apply its fixed configuration."""
        super().__init__(50, **kwargs)
