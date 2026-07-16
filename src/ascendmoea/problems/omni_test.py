"""Vectorized implementation of the omni_test problem."""

from __future__ import annotations

from .mmo_base import MMOProblem


class OmniTest(MMOProblem):
    """Implement the OmniTest vectorized optimization problem."""

    name = "Omni_test"

    def __init__(self, **kwargs):
        """Initialize the instance."""
        super().__init__(22, **kwargs)
