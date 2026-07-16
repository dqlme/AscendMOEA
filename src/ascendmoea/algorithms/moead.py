"""Implementation helpers for the moead algorithm."""

from __future__ import annotations

import math

import torch

from ..core import Algorithm, Problem
from ..utils import operator_ga, pairwise_dist, uniform_point


class MOEAD(Algorithm):
    """Optimize scalar subproblems over neighborhoods of reference weights."""

    name = "MOEAD"
    history_requires_snapshot = True

    def __init__(self, agg_type: int = 1, **kwargs):
        """Select the scalar aggregation function.

        Args:
            agg_type: ``1`` PBI, ``2`` weighted Tchebycheff, ``3`` normalized
                Tchebycheff, or ``4`` modified Tchebycheff.
            **kwargs: Arguments forwarded to :class:`~ascendmoea.Algorithm`.
        """
        super().__init__(**kwargs)
        self.agg_type = int(agg_type)
        if self.agg_type not in {1, 2, 3, 4}:
            raise ValueError("agg_type must be 1, 2, 3, or 4")

    def _solve(self, problem: Problem, output, should_stop, should_pause) -> None:
        """Execute the vectorized evolutionary loop."""
        torch.set_default_dtype(problem.dtype)
        weights, population_size = uniform_point(problem.N, problem.M, device=problem.device)
        weights = weights.to(dtype=problem.dtype, device=problem.device)
        problem.N = population_size
        neighbour_count = int(math.ceil(problem.N / 10))
        neighbours = torch.argsort(pairwise_dist(weights, weights), dim=1, stable=True)[
            :, :neighbour_count
        ]

        population = problem.initialization()
        ideal = population.objs.min(dim=0).values

        while self.not_terminated(problem, population, output, should_stop, should_pause):
            for subproblem in range(problem.N):
                order = torch.randperm(neighbour_count, device=problem.device)
                selected_neighbours = neighbours[subproblem, order]
                parent = population.decs[selected_neighbours[:2]]
                offspring = problem.evaluation(operator_ga(problem, parent, half=True))
                ideal = torch.minimum(ideal, offspring.objs[0])

                old_difference = population.objs[selected_neighbours] - ideal
                new_difference = offspring.objs[0] - ideal
                selected_weights = weights[selected_neighbours]
                if self.agg_type == 1:
                    weight_norm = torch.linalg.vector_norm(selected_weights, dim=1).clamp_min(
                        torch.finfo(problem.dtype).tiny
                    )
                    old_projection = torch.sum(
                        old_difference * selected_weights, dim=1
                    ) / weight_norm
                    new_projection = torch.sum(
                        new_difference.unsqueeze(0) * selected_weights, dim=1
                    ) / weight_norm
                    old_perpendicular = torch.sqrt(
                        torch.clamp(
                            torch.sum(old_difference.square(), dim=1) - old_projection.square(),
                            min=0,
                        )
                    )
                    new_perpendicular = torch.sqrt(
                        torch.clamp(
                            torch.sum(new_difference.square()) - new_projection.square(), min=0
                        )
                    )
                    old_value = old_projection + 5 * old_perpendicular
                    new_value = new_projection + 5 * new_perpendicular
                elif self.agg_type == 2:
                    old_value = torch.max(
                        torch.abs(old_difference) * selected_weights, dim=1
                    ).values
                    new_value = torch.max(
                        torch.abs(new_difference).unsqueeze(0) * selected_weights,
                        dim=1,
                    ).values
                elif self.agg_type == 3:
                    nadir = population.objs.max(dim=0).values
                    scale = (nadir - ideal).clamp_min(torch.finfo(problem.dtype).eps)
                    old_value = torch.max(
                        torch.abs(old_difference) / scale * selected_weights, dim=1
                    ).values
                    new_value = torch.max(
                        torch.abs(new_difference).unsqueeze(0) / scale * selected_weights,
                        dim=1,
                    ).values
                elif self.agg_type == 4:
                    old_value = torch.max(
                        torch.abs(old_difference) / selected_weights, dim=1
                    ).values
                    new_value = torch.max(
                        torch.abs(new_difference).unsqueeze(0) / selected_weights,
                        dim=1,
                    ).values
                else:
                    raise ValueError(f"Unsupported MOEA/D aggregation type: {self.agg_type}")

                replace = selected_neighbours[old_value >= new_value]
                if replace.numel() > 0:
                    population.decs[replace] = offspring.decs[0]
                    population.objs[replace] = offspring.objs[0]
                    population.cons[replace] = offspring.cons[0]
