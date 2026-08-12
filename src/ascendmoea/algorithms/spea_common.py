"""Implementation helpers for the spea_common algorithm."""

from __future__ import annotations

import math

import torch

from ..core import Population
from ..utils import pairwise_dist, sequential_euclidean_truncation


def subset_population(population: Population, index: torch.Tensor) -> Population:
    """Select population rows while preserving all population fields."""
    index = index.to(torch.long)
    return Population(
        population.decs[index],
        population.objs[index],
        population.cons[index],
    )


def cal_fitness(pop_obj: torch.Tensor, pop_con: torch.Tensor | None = None) -> torch.Tensor:
    """Calculate strength fitness with nearest-neighbor density."""
    n = pop_obj.shape[0]
    if n == 0:
        return torch.empty(0, dtype=pop_obj.dtype, device=pop_obj.device)
    cv = (
        torch.zeros(n, dtype=pop_obj.dtype, device=pop_obj.device)
        if pop_con is None
        else torch.clamp(pop_con, min=0).sum(dim=1)
    )
    cv_less = cv.unsqueeze(1) < cv.unsqueeze(0)
    cv_equal = cv.unsqueeze(1) == cv.unsqueeze(0)
    no_worse = pop_obj.unsqueeze(1) <= pop_obj.unsqueeze(0)
    better = pop_obj.unsqueeze(1) < pop_obj.unsqueeze(0)
    pareto = torch.all(no_worse, dim=2) & torch.any(better, dim=2)
    dominate = cv_less | (cv_equal & pareto)

    strength = dominate.sum(dim=1).to(pop_obj.dtype)
    raw = dominate.T.to(pop_obj.dtype) @ strength
    distance = pairwise_dist(pop_obj, pop_obj)
    distance.fill_diagonal_(torch.inf)
    neighbour = max(1, int(math.floor(math.sqrt(n))))
    density = 1 / (torch.kthvalue(distance, neighbour, dim=1).values + 2)
    return raw + density


def truncation(pop_obj: torch.Tensor, delete_count: int) -> torch.Tensor:
    """Choose candidates to remove by iterative distance comparison."""
    return sequential_euclidean_truncation(pop_obj, delete_count=delete_count)


def environmental_selection(
    population: Population,
    n: int,
    *,
    use_constraints: bool,
    sort_output: bool,
) -> tuple[Population, torch.Tensor]:
    """Select SPEA2 survivors and return their scalar fitness."""
    fitness = cal_fitness(population.objs, population.cons if use_constraints else None)
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
    population = subset_population(population, index)
    fitness = fitness[index]
    if sort_output:
        order = torch.argsort(fitness, stable=True)
        population = subset_population(population, order)
        fitness = fitness[order]
    return population, fitness
