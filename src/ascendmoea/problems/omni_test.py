"""Vectorized implementation of the omni_test problem."""

from __future__ import annotations

from .mmo_base import MMOProblem


class OmniTest(MMOProblem):
    """Define the OmniTest benchmark with batched tensor evaluation."""

    name = "Omni_test"

    def __init__(self, **kwargs):
        """Initialize the suite problem and apply its fixed configuration."""
        super().__init__(22, **kwargs)
