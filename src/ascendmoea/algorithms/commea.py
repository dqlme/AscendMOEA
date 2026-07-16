"""Implementation helpers for the commea algorithm."""

from __future__ import annotations

import math

import torch

from ..core import Algorithm, Problem
from ..utils import merge_pop, operator_ga, tournament_selection
from .mmo_common import _commea_environmental1, _commea_environmental2, _commea_fitness


class CoMMEA(Algorithm):
    """Cooperate between convergence and local decision-space niches."""

    name = "CoMMEA"

    def __init__(self, eps: float = 0.2, **kwargs):
        """Configure the minimum local-convergence neighborhood radius.

        Args:
            eps: Lower bound of the adaptive decision-space radius.
            **kwargs: Arguments forwarded to :class:`~ascendmoea.Algorithm`.
        """
        super().__init__(**kwargs)
        self.eps = float(eps)
        if self.eps <= 0:
            raise ValueError("eps must be positive")

    def _solve(self, problem: Problem, output, should_stop, should_pause) -> None:
        """Execute the vectorized evolutionary loop."""
        torch.set_default_dtype(problem.dtype)
        pop1 = problem.initialization()
        pop2 = problem.initialization()
        fitness1 = _commea_fitness(pop1, "Normal")
        fitness2 = _commea_fitness(pop2, "LocalC")
        while self.not_terminated(problem, pop2, output, should_stop, should_pause):
            mating1_count = int(math.floor(problem.N / 4 + 0.5))
            mating1 = tournament_selection(2, mating1_count, fitness1)
            mating2 = tournament_selection(2, problem.N, fitness2)
            parent1 = pop1.decs[mating1]
            parent2 = pop2.decs[mating2]
            off1 = problem.evaluation(operator_ga(problem, parent1))
            off2 = problem.evaluation(operator_ga(problem, parent2))
            off = merge_pop(off1, off2)
            evo_state = problem.FE / problem.max_fe
            cur_eps = max(-math.log2(max(1.5 * evo_state, 1e-12)), self.eps)
            pop1, fitness1 = _commea_environmental1(
                merge_pop(pop1, off), problem.N, "Normal", evo_state
            )
            pop2, fitness2 = _commea_environmental2(
                merge_pop(pop2, off), problem.N, "LocalC", cur_eps
            )
