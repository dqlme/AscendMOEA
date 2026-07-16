"""Implementation helpers for the cmegl algorithm."""

from __future__ import annotations

import torch

from ..core import Algorithm, Population, Problem
from ..utils import merge_pop, operator_ga, tournament_selection
from .spea_common import cal_fitness, environmental_selection, subset_population, truncation


def _constraint_violation(population: Population) -> torch.Tensor:
    """Compute constraint violation."""
    return torch.clamp(population.cons, min=0).sum(dim=1)


def _local_environmental_selection(
    population: Population, n: int, threshold: float
) -> tuple[Population, torch.Tensor]:
    """Compute local environmental selection."""
    cv = _constraint_violation(population)
    accepted = torch.where(cv <= threshold)[0]
    local_population = subset_population(population, accepted)
    if accepted.numel() == 0:
        return local_population, torch.empty(
            0, dtype=population.objs.dtype, device=population.objs.device
        )

    local_cv = _constraint_violation(local_population)
    fitness = cal_fitness(torch.cat([local_population.objs, local_cv.unsqueeze(1)], dim=1))
    if local_population.objs.shape[0] <= n:
        order = torch.argsort(fitness, stable=True)
        return subset_population(local_population, order), fitness[order]

    selected = fitness < 1
    selected_count = int(selected.sum().item())
    if selected_count <= n:
        rank = torch.argsort(fitness, stable=True)
        selected[rank[:n]] = True
    else:
        active = torch.where(selected)[0]
        deleted = truncation(local_population.objs[active], selected_count - n)
        selected[active[deleted]] = False
    index = torch.where(selected)[0]
    local_population = subset_population(local_population, index)
    fitness = fitness[index]
    order = torch.argsort(fitness, stable=True)
    return subset_population(local_population, order), fitness[order]


class CMEGL(Algorithm):
    """Implement the CMEGL multi-objective evolutionary algorithm."""

    name = "CMEGL"

    def _solve(self, problem: Problem, output, should_stop, should_pause) -> None:
        """Execute the vectorized evolutionary loop."""
        torch.set_default_dtype(problem.dtype)
        population1 = problem.initialization()
        fitness1 = cal_fitness(population1.objs, population1.cons)
        population2 = problem.initialization()
        fitness2 = cal_fitness(population2.objs)
        population3 = problem.initialization()
        fitness3 = cal_fitness(population3.objs, population3.cons)

        cv = _constraint_violation(population1)
        positive_cv = cv[cv > 0]
        threshold = float(positive_cv.mean().item()) if positive_cv.numel() else 0.0
        generation = 0
        global_finished = False
        objective_std_history: list[torch.Tensor] = []

        while self.not_terminated(problem, population1, output, should_stop, should_pause):
            generation += 1
            if not global_finished:
                objective_std_history.append(torch.std(population2.objs, dim=0, correction=1))
                if len(objective_std_history) > 101:
                    objective_std_history.pop(0)
                if generation > 100 and torch.all(
                    torch.std(torch.stack(objective_std_history), dim=0, correction=1) < 0.5
                ):
                    global_finished = True

            mating1 = tournament_selection(2, problem.N, fitness1)
            offspring1 = problem.evaluation(
                operator_ga(problem, population1.decs[mating1], half=True)
            )

            offspring2 = None
            if not global_finished:
                mating2 = tournament_selection(2, problem.N, fitness2)
                offspring2 = problem.evaluation(
                    operator_ga(problem, population2.decs[mating2], half=True)
                )

            offspring3 = None
            if population3.objs.shape[0] > 1:
                mating_count = min(population3.objs.shape[0], problem.N // 2)
                mating3 = tournament_selection(2, mating_count, fitness3)
                offspring3 = problem.evaluation(operator_ga(problem, population3.decs[mating3]))

            auxiliary_offspring = [
                offspring for offspring in (offspring2, offspring3) if offspring is not None
            ]
            if auxiliary_offspring:
                population1, fitness1 = environmental_selection(
                    merge_pop(population1, *auxiliary_offspring),
                    problem.N,
                    use_constraints=True,
                    sort_output=True,
                )
            population1, fitness1 = environmental_selection(
                merge_pop(population1, offspring1),
                problem.N,
                use_constraints=True,
                sort_output=True,
            )

            if not global_finished:
                population2, fitness2 = environmental_selection(
                    merge_pop(population2, offspring1, *auxiliary_offspring),
                    problem.N,
                    use_constraints=False,
                    sort_output=True,
                )

            population3, fitness3 = _local_environmental_selection(
                merge_pop(population3, offspring1, *auxiliary_offspring),
                problem.N,
                threshold,
            )
            offspring_cv = _constraint_violation(offspring1)
            positive_cv = offspring_cv[offspring_cv > 0]
            threshold = float(positive_cv.mean().item()) if positive_cv.numel() else 0.0
