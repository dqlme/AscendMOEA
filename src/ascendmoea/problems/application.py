"""Vectorized implementation of the application problem."""

from __future__ import annotations

from .mmo_base import MMOProblem


class ApplicationProblem(MMOProblem):
    """Implement the ApplicationProblem vectorized optimization problem."""

    name = "application"

    def __init__(self, **kwargs):
        """Initialize the instance."""
        super().__init__(72, **kwargs)
