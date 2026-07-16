"""Implementation helpers for the cmoead algorithm."""

from __future__ import annotations

import math

import torch

from ..core import Algorithm, Problem
from ..utils import operator_ga, pairwise_dist, uniform_point


class CMOEAD(Algorithm):
    """Implement the CMOEAD multi-objective evolutionary algorithm."""

    name = "CMOEAD"
    history_requires_snapshot = True

    def _solve(self, problem: Problem, output, should_stop, should_pause) -> None:
        """Execute the vectorized evolutionary loop."""
        torch.set_default_dtype(problem.dtype)
        weights, population_size = uniform_point(problem.N, problem.M, device=problem.device)
        weights = weights.to(dtype=problem.dtype, device=problem.device)
        problem.N = population_size
        neighbour_count = int(math.ceil(problem.N / 10))
        replacement_limit = int(math.ceil(problem.N / 100))
        neighbours = torch.argsort(pairwise_dist(weights, weights), dim=1, stable=True)[
            :, :neighbour_count
        ]

        population = problem.initialization()
        ideal = population.objs.min(dim=0).values

        while self.not_terminated(problem, population, output, should_stop, should_pause):
            for subproblem in range(problem.N):
                if torch.rand((), dtype=problem.dtype, device=problem.device) < 0.9:
                    selected = neighbours[
                        subproblem,
                        torch.randperm(neighbour_count, device=problem.device),
                    ]
                else:
                    selected = torch.randperm(problem.N, device=problem.device)
                offspring = problem.evaluation(
                    operator_ga(problem, population.decs[selected[:2]], half=True)
                )
                ideal = torch.minimum(ideal, offspring.objs[0])

                offspring_cv = torch.clamp(offspring.cons[0], min=0).sum()
                selected_cv = torch.clamp(population.cons[selected], min=0).sum(dim=1)
                difference = population.objs[selected] - ideal
                offspring_difference = offspring.objs[0] - ideal
                selected_weights = weights[selected]
                weight_norm = torch.linalg.vector_norm(selected_weights, dim=1)
                old_norm = torch.linalg.vector_norm(difference, dim=1)
                new_norm = torch.linalg.vector_norm(offspring_difference)
                old_cosine = (
                    torch.sum(difference * selected_weights, dim=1) / weight_norm / old_norm
                )
                new_cosine = (
                    torch.sum(offspring_difference.unsqueeze(0) * selected_weights, dim=1)
                    / weight_norm
                    / new_norm
                )
                old_value = old_norm * old_cosine + 5 * old_norm * torch.sqrt(1 - old_cosine**2)
                new_value = new_norm * new_cosine + 5 * new_norm * torch.sqrt(1 - new_cosine**2)
                replace_mask = ((old_value >= new_value) & (selected_cv == offspring_cv)) | (
                    selected_cv > offspring_cv
                )
                replace = selected[torch.where(replace_mask)[0][:replacement_limit]]
                if replace.numel() > 0:
                    population.decs[replace] = offspring.decs[0]
                    population.objs[replace] = offspring.objs[0]
                    population.cons[replace] = offspring.cons[0]
