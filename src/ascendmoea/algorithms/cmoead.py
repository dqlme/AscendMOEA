"""Implementation helpers for the cmoead algorithm."""

from __future__ import annotations

import math

import torch

from ..core import Algorithm, Problem
from ..utils import operator_ga, pairwise_dist, uniform_point


class CMOEAD(Algorithm):
    """Apply decomposition and constraint violation to neighborhood replacement."""

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
                weight_norm = torch.linalg.vector_norm(selected_weights, dim=1).clamp_min(
                    torch.finfo(problem.dtype).tiny
                )
                old_projection = torch.sum(difference * selected_weights, dim=1) / weight_norm
                new_projection = torch.sum(
                    offspring_difference.unsqueeze(0) * selected_weights, dim=1
                ) / weight_norm
                old_perpendicular = torch.sqrt(
                    torch.clamp(
                        torch.sum(difference.square(), dim=1) - old_projection.square(), min=0
                    )
                )
                new_perpendicular = torch.sqrt(
                    torch.clamp(
                        torch.sum(offspring_difference.square()) - new_projection.square(), min=0
                    )
                )
                old_value = old_projection + 5 * old_perpendicular
                new_value = new_projection + 5 * new_perpendicular
                replace_mask = ((old_value >= new_value) & (selected_cv == offspring_cv)) | (
                    selected_cv > offspring_cv
                )
                replace = selected[torch.where(replace_mask)[0][:replacement_limit]]
                if replace.numel() > 0:
                    population.decs[replace] = offspring.decs[0]
                    population.objs[replace] = offspring.objs[0]
                    population.cons[replace] = offspring.cons[0]
