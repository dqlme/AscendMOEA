"""Implementation helpers for the sparse_ea algorithm."""

from __future__ import annotations

import torch

from ..core import Algorithm, Problem
from ..utils import merge_pop, operator_ga, tournament_selection
from .sparse_common import (
    environmental_selection,
    initial_mask,
    random_decisions,
    sparse_mask_crossover,
    sparse_mask_mutation,
    variable_fitness,
)


class SparseEA(Algorithm):
    """Evolve decision values and sparse activation masks jointly."""

    name = "SparseEA"

    @staticmethod
    def _operator(
        problem: Problem,
        parent_dec: torch.Tensor,
        parent_mask: torch.Tensor,
        fitness: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute operator."""
        half = parent_dec.shape[0] // 2
        mask = sparse_mask_crossover(parent_mask[:half], parent_mask[half : 2 * half], fitness)
        mask = sparse_mask_mutation(mask, fitness)
        if torch.any(problem.encoding != 4):
            dec = operator_ga(problem, parent_dec, half=True)
            dec[:, problem.encoding == 4] = 1
        else:
            dec = torch.ones((half, problem.D), dtype=problem.dtype, device=problem.device)
        return dec, mask

    def _solve(self, problem: Problem, output, should_stop, should_pause) -> None:
        """Execute the vectorized evolutionary loop."""
        torch.set_default_dtype(problem.dtype)
        fitness, temp_pop, temp_dec, temp_mask = variable_fitness(problem, keep_all_samples=True)

        dec = random_decisions(problem, problem.N)
        mask = initial_mask(problem.N, problem.D, fitness)
        population = problem.evaluation(dec * mask.to(dec.dtype))
        state = environmental_selection(
            merge_pop(population, temp_pop),
            torch.cat([dec, temp_dec], dim=0),
            torch.cat([mask, temp_mask], dim=0),
            problem.N,
        )

        while self.not_terminated(problem, state.population, output, should_stop, should_pause):
            mating = tournament_selection(2, 2 * problem.N, state.front_no, -state.crowd_dis)
            off_dec, off_mask = self._operator(
                problem, state.dec[mating], state.mask[mating], fitness
            )
            offspring = problem.evaluation(off_dec * off_mask.to(off_dec.dtype))
            state = environmental_selection(
                merge_pop(state.population, offspring),
                torch.cat([state.dec, off_dec], dim=0),
                torch.cat([state.mask, off_mask], dim=0),
                problem.N,
            )
