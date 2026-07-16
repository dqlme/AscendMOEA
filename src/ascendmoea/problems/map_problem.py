"""Vectorized implementation of the map_problem problem."""

from __future__ import annotations

from .mmo_base import MMOProblem


class MapProblem(MMOProblem):
    """Define the MapProblem benchmark with batched tensor evaluation."""

    name = "map"

    def __init__(self, **kwargs):
        """Initialize the suite problem and apply its fixed configuration."""
        super().__init__(71, **kwargs)
