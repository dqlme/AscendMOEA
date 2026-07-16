"""Vectorized implementation of the mmmop3d problem."""

from __future__ import annotations

from .mmo_base import MMOProblem


class MMMOP3D(MMOProblem):
    """Implement the MMMOP3D vectorized optimization problem."""

    name = "MMMOP3D"

    def __init__(self, **kwargs):
        """Initialize the instance."""
        super().__init__(50, **kwargs)
