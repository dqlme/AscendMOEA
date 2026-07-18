"""Implementation helpers for the cmoeams algorithm."""

from __future__ import annotations

import math

import torch

from ..core import Algorithm, Population, Problem
from ..utils import (
    merge_pop,
    operator_de,
    operator_ga,
    sequential_distance_truncation,
    tournament_selection,
)


def _subset(population: Population, index: torch.Tensor) -> Population:
    """Select population rows while preserving all population fields."""
    index = index.to(torch.long)
    return Population(
        population.decs[index],
        population.objs[index],
        population.cons[index],
    )


def _cal_cv(constraints: torch.Tensor) -> torch.Tensor:
    """Aggregate positive constraint violations for each individual."""
    violation = torch.clamp(constraints, min=0)
    maximum = violation.max(dim=0).values
    normalized = torch.where(maximum > 0, violation / maximum, torch.zeros_like(violation))
    return normalized.mean(dim=1)


def _cosine_distance(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
    """Return pairwise cosine distances with zero-vector protection."""
    tiny = torch.finfo(first.dtype).tiny
    first = first / torch.linalg.vector_norm(first, dim=1, keepdim=True).clamp_min(tiny)
    second = second / torch.linalg.vector_norm(second, dim=1, keepdim=True).clamp_min(tiny)
    return torch.clamp(1 - first @ second.T, min=0, max=2)


def _cal_sde(pop_obj: torch.Tensor) -> torch.Tensor:
    """Compute shift-based density estimation for every objective vector."""
    n = pop_obj.shape[0]
    minimum = pop_obj.min(dim=0).values
    maximum = pop_obj.max(dim=0).values
    span = (maximum - minimum).clamp_min(torch.finfo(pop_obj.dtype).eps)
    normalized = (pop_obj - minimum) / span
    shifted = torch.maximum(normalized.unsqueeze(1), normalized.unsqueeze(0))
    distance = torch.linalg.vector_norm(normalized.unsqueeze(1) - shifted, dim=2)
    neighbour = min(n, int(math.floor(math.sqrt(n))) + 1)
    return 1 / (torch.kthvalue(distance, neighbour, dim=1).values + 2)


def _cal_fitness(pop_obj: torch.Tensor, cv: torch.Tensor | None = None) -> torch.Tensor:
    """Calculate stage-dependent fitness from objectives and constraints."""
    n = pop_obj.shape[0]
    if cv is None:
        cv = torch.zeros(n, dtype=pop_obj.dtype, device=pop_obj.device)
    else:
        cv = cv.reshape(-1)
    cv_less = cv.unsqueeze(1) < cv.unsqueeze(0)
    cv_equal = cv.unsqueeze(1) == cv.unsqueeze(0)
    no_worse = pop_obj.unsqueeze(1) <= pop_obj.unsqueeze(0)
    better = pop_obj.unsqueeze(1) < pop_obj.unsqueeze(0)
    dominate = cv_less | (cv_equal & torch.all(no_worse, dim=2) & torch.any(better, dim=2))
    strength = dominate.sum(dim=1).to(pop_obj.dtype)
    raw = dominate.T.to(pop_obj.dtype) @ strength
    distance = _cosine_distance(torch.real(pop_obj), torch.real(pop_obj))
    distance.fill_diagonal_(torch.inf)
    neighbour = max(1, int(math.floor(math.sqrt(n))))
    density = 1 / (torch.kthvalue(distance, neighbour, dim=1).values + 2)
    return raw + density


def _truncation(pop_obj: torch.Tensor, delete_count: int) -> torch.Tensor:
    """Remove crowded candidates by sequential distance comparison."""
    distance = _cosine_distance(pop_obj, pop_obj)
    return sequential_distance_truncation(distance, delete_count)


def _environmental_selection(
    population: Population, fitness: torch.Tensor, n: int
) -> tuple[Population, torch.Tensor]:
    """Select survivors for the current feasibility-search stage."""
    selected = fitness < 1
    selected_count = int(selected.sum().item())
    if selected_count < n:
        rank = torch.argsort(fitness, stable=True)
        selected[rank[:n]] = True
    elif selected_count > n:
        active = torch.where(selected)[0]
        deleted = _truncation(population.objs[active], selected_count - n)
        selected[active[deleted]] = False
    return _subset(population, torch.where(selected)[0]), fitness[selected]


class CMOEAMS(Algorithm):
    """Balance objective optimization and feasibility across multiple stages."""

    name = "CMOEAMS"

    def __init__(self, op_type: int = 1, lam: float = 0.5, **kwargs):
        """Configure variation and the feasible-ratio stage threshold.

        Args:
            op_type: Variation operator, where ``1`` is GA and ``2`` is DE.
            lam: Feasible-population ratio required to enter the final stage.
            **kwargs: Arguments forwarded to :class:`~ascendmoea.Algorithm`.
        """
        super().__init__(**kwargs)
        self.op_type = int(op_type)
        self.lam = float(lam)
        if self.op_type not in {1, 2}:
            raise ValueError("op_type must be 1 (GA) or 2 (DE)")
        if not 0 <= self.lam <= 1:
            raise ValueError("lam must be between zero and one")

    def _solve(self, problem: Problem, output, should_stop, should_pause) -> None:
        """Execute the vectorized evolutionary loop."""
        torch.set_default_dtype(problem.dtype)
        population = problem.initialization()
        cv = _cal_cv(population.cons)
        fitness = _cal_fitness(torch.stack([_cal_sde(population.objs), cv], dim=1))

        while self.not_terminated(problem, population, output, should_stop, should_pause):
            if self.op_type == 1:
                mating = tournament_selection(2, problem.N, fitness)
                offspring = problem.evaluation(operator_ga(problem, population.decs[mating]))
            elif self.op_type == 2:
                mating1 = tournament_selection(2, problem.N, fitness)
                mating2 = tournament_selection(2, problem.N, fitness)
                offspring = problem.evaluation(
                    operator_de(
                        problem,
                        population.decs,
                        population.decs[mating1],
                        population.decs[mating2],
                    )
                )
            else:
                raise ValueError(f"Unsupported CMOEA-MS operator type: {self.op_type}")

            merged = merge_pop(population, offspring)
            cv = _cal_cv(merged.cons)
            if (
                float((cv <= 0).to(problem.dtype).mean().item()) > self.lam
                and problem.FE >= 0.1 * problem.max_fe
            ):
                fitness = _cal_fitness(merged.objs, cv)
            else:
                fitness = _cal_fitness(torch.stack([_cal_sde(merged.objs), cv], dim=1))
            population, fitness = _environmental_selection(merged, fitness, problem.N)
