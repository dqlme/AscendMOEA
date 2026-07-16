"""Tensorized crossover, mutation, and differential-evolution operators."""

from ..encoded_ga import operator_ga_encoded
from ..utils import operator_de, operator_ga

genetic_operator = operator_ga
differential_evolution_operator = operator_de

__all__ = [
    "differential_evolution_operator",
    "genetic_operator",
    "operator_de",
    "operator_ga",
    "operator_ga_encoded",
]
