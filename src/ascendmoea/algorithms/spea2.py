"""Implementation helpers for the spea2 algorithm."""

from __future__ import annotations

import torch

from ..core import Algorithm, Problem
from ..utils import merge_pop, operator_ga, tournament_selection
from .spea_common import cal_fitness, environmental_selection


class SPEA2(Algorithm):
    """Implement the SPEA2 multi-objective evolutionary algorithm."""

    name = "SPEA2"

    def _solve(self, problem: Problem, output, should_stop, should_pause) -> None:
        """Execute the vectorized evolutionary loop."""
        torch.set_default_dtype(problem.dtype)
        population = problem.initialization()
        fitness = cal_fitness(population.objs)

        while self.not_terminated(problem, population, output, should_stop, should_pause):
            mating = tournament_selection(2, problem.N, fitness)
            offspring = problem.evaluation(operator_ga(problem, population.decs[mating]))
            population, fitness = environmental_selection(
                merge_pop(population, offspring),
                problem.N,
                use_constraints=False,
                sort_output=False,
            )
