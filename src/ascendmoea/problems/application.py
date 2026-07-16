"""Vectorized implementation of the application problem."""

from __future__ import annotations

from .mmo_base import MMOProblem


class ApplicationProblem(MMOProblem):
    """Define the ApplicationProblem benchmark with batched tensor evaluation."""

    name = "application"

    def __init__(self, **kwargs):
        """Initialize the suite problem and apply its fixed configuration."""
        super().__init__(72, **kwargs)
