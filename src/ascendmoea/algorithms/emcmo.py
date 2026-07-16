"""Implementation helpers for the emcmo algorithm."""

from __future__ import annotations

import math

import torch

from ..core import Algorithm, Population, Problem
from ..utils import merge_pop, operator_ga, pairwise_dist, tournament_selection
from .spea_common import truncation


def _subset(population: Population, index: torch.Tensor) -> Population:
    """Compute subset."""
    index = index.to(torch.long)
    return Population(
        population.decs[index],
        population.objs[index],
        population.cons[index],
    )


def _cal_fitness(pop_obj: torch.Tensor, pop_con: torch.Tensor | None) -> torch.Tensor:
    """Compute cal fitness."""
    n = pop_obj.shape[0]
    cv = (
        torch.zeros(n, dtype=pop_obj.dtype, device=pop_obj.device)
        if pop_con is None
        else torch.clamp(pop_con, min=0).sum(dim=1)
    )
    cv_less = cv.unsqueeze(1) < cv.unsqueeze(0)
    cv_equal = cv.unsqueeze(1) == cv.unsqueeze(0)
    no_worse = pop_obj.unsqueeze(1) <= pop_obj.unsqueeze(0)
    better = pop_obj.unsqueeze(1) < pop_obj.unsqueeze(0)
    pareto_dominates = torch.all(no_worse, dim=2) & torch.any(better, dim=2)
    dominates = cv_less | (cv_equal & pareto_dominates)

    strength = dominates.sum(dim=1).to(pop_obj.dtype)
    raw_fitness = dominates.transpose(0, 1).to(pop_obj.dtype) @ strength
    distance = pairwise_dist(pop_obj, pop_obj)
    distance.fill_diagonal_(float("inf"))
    neighbour = max(1, math.floor(math.sqrt(n)))
    density = 1 / (torch.kthvalue(distance, neighbour, dim=1).values + 2)
    return raw_fitness + density


def _environmental_selection(
    population: Population, n: int, is_origin: bool
) -> tuple[Population, torch.Tensor, torch.Tensor]:
    """Compute environmental selection."""
    fitness = _cal_fitness(population.objs, population.cons if is_origin else None)
    selected = fitness < 1
    selected_count = int(selected.sum().item())
    if selected_count < n:
        rank = torch.argsort(fitness, stable=True)
        selected[rank[:n]] = True
    elif selected_count > n:
        active = torch.where(selected)[0]
        deleted = truncation(population.objs[active], selected_count - n)
        selected[active[deleted]] = False

    index = torch.where(selected)[0]
    next_population = _subset(population, index)
    next_fitness = fitness[index]
    order = torch.argsort(next_fitness, stable=True)
    return _subset(next_population, order), next_fitness[order], selected


class EMCMO(Algorithm):
    """Implement the EMCMO multi-objective evolutionary algorithm."""

    name = "EMCMO"

    def _solve(self, problem: Problem, output, should_stop, should_pause) -> None:
        """Execute the vectorized evolutionary loop."""
        torch.set_default_dtype(problem.dtype)
        population1 = problem.initialization()
        population2 = problem.initialization()
        fitness1 = _cal_fitness(population1.objs, population1.cons)
        fitness2 = _cal_fitness(population2.objs, None)
        transfer_state = 0

        while self.not_terminated(problem, population1, output, should_stop, should_pause):
            if transfer_state == 0:
                index1 = torch.randint(0, problem.N, (problem.N,), device=problem.device)
                index2 = torch.randint(0, problem.N, (problem.N,), device=problem.device)
                offspring1 = problem.evaluation(
                    operator_ga(problem, population1.decs[index1], half=True)
                )
                offspring2 = problem.evaluation(
                    operator_ga(problem, population2.decs[index2], half=True)
                )
                population1, fitness1, _ = _environmental_selection(
                    merge_pop(population1, offspring1, offspring2), problem.N, True
                )
                population2, fitness2, _ = _environmental_selection(
                    merge_pop(population2, offspring2, offspring1), problem.N, False
                )
                if problem.FE / problem.max_fe >= 0.2:
                    transfer_state = 1
                continue

            mating1 = tournament_selection(2, problem.N, fitness1)
            mating2 = tournament_selection(2, problem.N, fitness2)
            offspring1 = problem.evaluation(
                operator_ga(problem, population1.decs[mating1], half=True)
            )
            offspring2 = problem.evaluation(
                operator_ga(problem, population2.decs[mating2], half=True)
            )

            _, _, selected1 = _environmental_selection(
                merge_pop(population2, offspring2), problem.N, True
            )
            success1 = (
                selected1[: problem.N].sum().item() / 100 - selected1[problem.N :].sum().item() / 50
            )
            _, _, selected2 = _environmental_selection(
                merge_pop(population1, offspring1), problem.N, False
            )
            success2 = (
                selected2[: problem.N].sum().item() / 100 - selected2[problem.N :].sum().item() / 50
            )

            if success1 > 0:
                random_index = torch.randperm(problem.N, device=problem.device)[: problem.N // 2]
                population1, fitness1, _ = _environmental_selection(
                    merge_pop(
                        population1,
                        offspring1,
                        _subset(population2, random_index),
                    ),
                    problem.N,
                    True,
                )
            else:
                population1, fitness1, _ = _environmental_selection(
                    merge_pop(population1, offspring1, offspring2), problem.N, True
                )

            if success2 > 0:
                random_index = torch.randperm(problem.N, device=problem.device)[: problem.N // 2]
                population2, fitness2, _ = _environmental_selection(
                    merge_pop(
                        population2,
                        offspring2,
                        _subset(population1, random_index),
                    ),
                    problem.N,
                    False,
                )
            else:
                population2, fitness2, _ = _environmental_selection(
                    merge_pop(population2, offspring2, offspring1), problem.N, False
                )
