"""Implementation helpers for the sparse_ea2 algorithm."""

from __future__ import annotations

import torch

from ..core import Algorithm, Problem
from ..utils import merge_pop, tournament_selection
from .sparse_common import (
    environmental_selection,
    initial_mask,
    polynomial_mutation_at_sites,
    random_decisions,
    sbx_first_child,
    sparse_mask_crossover,
    sparse_mask_mutation,
    variable_fitness,
)


class SparseEA2(Algorithm):
    """Enhance sparse mask evolution with grouped decision-variable variation."""

    name = "SparseEA2"
    number_of_groups = 4

    @classmethod
    def _grouped_real_operator(
        cls,
        problem: Problem,
        parent1: torch.Tensor,
        parent2: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute grouped real operator."""
        offspring = sbx_first_child(parent1, parent2)
        n, d = offspring.shape
        order = torch.argsort(offspring, dim=1, stable=True)
        variables_per_group = d // cls.number_of_groups
        if variables_per_group == 0:
            rank_group = torch.full(
                (d,), cls.number_of_groups - 1, dtype=torch.long, device=offspring.device
            )
        else:
            rank_group = torch.div(
                torch.arange(d, device=offspring.device),
                variables_per_group,
                rounding_mode="floor",
            ).clamp(max=cls.number_of_groups - 1)
        groups = torch.empty((n, d), dtype=torch.long, device=offspring.device)
        groups.scatter_(1, order, rank_group.view(1, d).expand(n, d))
        chosen = torch.randint(0, cls.number_of_groups, (n,), device=offspring.device)
        sites = groups == chosen.unsqueeze(1)
        mu = torch.rand((n, 1), dtype=offspring.dtype, device=offspring.device)
        offspring = polynomial_mutation_at_sites(offspring, problem.lower, problem.upper, sites, mu)
        return offspring, groups, chosen

    @classmethod
    def _operator(
        cls,
        problem: Problem,
        parent_dec: torch.Tensor,
        parent_mask: torch.Tensor,
        fitness: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute operator."""
        half = parent_dec.shape[0] // 2
        parent1_dec = parent_dec[:half]
        parent2_dec = parent_dec[half : 2 * half]
        parent1_mask = parent_mask[:half]
        parent2_mask = parent_mask[half : 2 * half]

        if torch.any(problem.encoding != 4):
            dec, groups, chosen = cls._grouped_real_operator(problem, parent1_dec, parent2_dec)
            dec[:, problem.encoding == 4] = 1
        else:
            dec = torch.ones_like(parent1_dec)
            groups = torch.zeros_like(parent1_mask, dtype=torch.long)
            chosen = torch.zeros(half, dtype=torch.long, device=problem.device)

        mask = sparse_mask_crossover(parent1_mask, parent2_mask, fitness)
        if torch.any(problem.encoding != 4):
            mask = sparse_mask_mutation(mask, fitness, groups == chosen.unsqueeze(1))
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
