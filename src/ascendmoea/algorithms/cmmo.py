"""Implementation helpers for the cmmo algorithm."""

from __future__ import annotations

import math

import torch

from ..core import Algorithm, Problem
from ..utils import merge_pop, operator_ga, tournament_selection
from .mmo_common import (
    _cmmo_dec_epsilon_fitness,
    _cmmo_fitness,
    _spea2_environmental,
)


class CMMO(Algorithm):
    """Implement the CMMO multi-objective evolutionary algorithm."""

    name = "CMMO"

    def __init__(self, eta: float = 0.2, tao: float = 0.1, theta: float = 0.1, **kwargs):
        """Initialize the instance."""
        super().__init__(**kwargs)
        self.eta = float(eta)
        self.tao = float(tao)
        self.theta = float(theta)

    def _solve(self, problem: Problem, output, should_stop, should_pause) -> None:
        """Execute the vectorized evolutionary loop."""
        torch.set_default_dtype(problem.dtype)
        pop1 = problem.initialization()
        pop2 = problem.initialization()
        generation_count = int(math.ceil(problem.max_fe / (2 * problem.N)))
        generation = 1
        control_generation = (1 - self.eta) * generation_count
        epsilon0 = torch.sum(torch.max(pop2.objs, dim=0).values)
        epsilon_k = epsilon0.clone()
        d_dec, d_pop, fitness1 = _cmmo_fitness(pop1)
        fitness2, d = _cmmo_dec_epsilon_fitness(pop2, epsilon_k)
        while self.not_terminated(problem, pop1, output, should_stop, should_pause):
            if generation < self.eta * generation_count:
                epsilon_k = torch.zeros((), dtype=problem.dtype, device=problem.device)
            elif generation < control_generation:
                epsilon_k = (1 - self.tao) * epsilon_k
                if float(epsilon_k.item()) <= self.theta:
                    epsilon_k = (generation / generation_count) * epsilon0
            mating1 = tournament_selection(2, problem.N, d_dec, d_pop, fitness1)
            mating2 = tournament_selection(2, problem.N, d, fitness2)
            parent1 = pop1.decs[mating1]
            parent2 = pop2.decs[mating2]
            off1 = problem.evaluation(operator_ga(problem, parent1, half=True))
            off2 = problem.evaluation(operator_ga(problem, parent2, half=True))
            off = merge_pop(off1, off2)
            pop1, fitness1, d_dec, d_pop = _spea2_environmental(merge_pop(pop1, off), problem.N)
            pop2, fitness2, d, _ = _spea2_environmental(
                merge_pop(pop2, off), problem.N, float(epsilon_k.item())
            )
            generation = int(math.ceil(problem.FE / (2 * problem.N)))
            epsilon0 = torch.minimum(torch.max(torch.sum(pop2.objs, dim=1)), epsilon0)
