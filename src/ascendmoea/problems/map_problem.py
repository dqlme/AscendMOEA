"""Vectorized implementation of the map_problem problem."""

from __future__ import annotations

from .mmo_base import MMOProblem


class MapProblem(MMOProblem):
    """Implement the MapProblem vectorized optimization problem."""

    name = "map"

    def __init__(self, **kwargs):
        """Initialize the instance."""
        super().__init__(71, **kwargs)
