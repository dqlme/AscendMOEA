"""Implementation helpers for the bico algorithm."""

from __future__ import annotations

import torch

from ..core import Algorithm, Population, Problem
from ..utils import (
    merge_pop,
    nd_sort,
    operator_ga,
    pairwise_dist,
    sequential_distance_truncation,
    tournament_selection,
)
from .sparse_common import subset_population


def _constraint_violation(population: Population) -> torch.Tensor:
    """Aggregate positive constraint violations for each individual."""
    return torch.clamp(population.cons, min=0).sum(dim=1)


def _cosine_similarity(points: torch.Tensor) -> torch.Tensor:
    """Return self-cosine similarities without zero-vector failures."""
    norm = torch.linalg.vector_norm(points, dim=1, keepdim=True)
    points = points / norm.clamp_min(torch.finfo(points.dtype).tiny)
    cosine = torch.clamp(points @ points.T, min=-1, max=1)
    return cosine * (1 - torch.eye(points.shape[0], dtype=points.dtype, device=points.device))


def _distance_truncation(pop_obj: torch.Tensor, delete_count: int) -> torch.Tensor:
    """Retain diverse candidates by sequential nearest-distance deletion."""
    distance = pairwise_dist(pop_obj, pop_obj)
    return sequential_distance_truncation(distance, delete_count)


class BiCo(Algorithm):
    """Co-evolve complementary populations for constrained optimization."""

    name = "BiCo"

    @staticmethod
    def _update_archive(population: Population, n: int) -> Population:
        """Update the infeasible archive with convergence and diversity pressure."""
        cv = _constraint_violation(population)
        front_no, _ = nd_sort(torch.cat([population.objs, cv.unsqueeze(1)], dim=1), n_sort=1)
        selected = (front_no == 1) & (cv > 0)
        archive = subset_population(population, torch.where(selected)[0])
        if archive.objs.shape[0] < n:
            return archive

        archive_cv = _constraint_violation(archive)
        zmax = archive.objs.max(dim=0).values
        zmin = archive.objs.min(dim=0).values
        normalized = (archive.objs - zmax) / (zmin - zmax - 1e-10)
        cosine = _cosine_similarity(normalized)
        deleted = torch.zeros(archive.objs.shape[0], dtype=torch.bool, device=archive.objs.device)
        for _ in range(archive.objs.shape[0] - n):
            maximum = cosine.max()
            candidates = torch.nonzero(cosine == maximum, as_tuple=False)
            pair = candidates[torch.randint(0, candidates.shape[0], (), device=archive.objs.device)]
            first, second = pair[0], pair[1]
            delete_first = (archive_cv[first] > archive_cv[second]) | (
                (archive_cv[first] == archive_cv[second])
                & (torch.rand((), dtype=archive.objs.dtype, device=archive.objs.device) < 0.5)
            )
            target = torch.where(delete_first, first, second)
            deleted[target] = True
            cosine[target, :] = 0
            cosine[:, target] = 0
        return subset_population(archive, torch.where(~deleted)[0])

    @staticmethod
    def _mating_selection(population: Population, archive: Population, n: int) -> Population:
        """Select parent indices from the main population and archive."""
        if archive.objs.shape[0] < n:
            selected = tournament_selection(2, n, -_constraint_violation(population))
            return subset_population(population, selected)

        all_population = merge_pop(population, archive)
        zmin = all_population.objs.min(dim=0).values
        normalized = (all_population.objs - zmin) / (
            all_population.objs.max(dim=0).values - zmin + 1e-10
        ) + 1e-10
        cosine = _cosine_similarity(normalized)
        sorted_negative_cosine = torch.sort(-cosine, dim=1).values
        rank = torch.arange(sorted_negative_cosine.shape[0], device=cosine.device)
        for column in range(sorted_negative_cosine.shape[1] - 1, -1, -1):
            rank = rank[torch.argsort(sorted_negative_cosine[rank, column], stable=True)]

        population_size = population.objs.shape[0]
        angle_population = rank[:population_size]
        angle_archive = rank[population_size:]
        population_index = torch.randint(0, population_size, (n,), device=population.objs.device)
        archive_index = torch.randint(0, archive.objs.shape[0], (n,), device=population.objs.device)
        choose_population = torch.empty(n, dtype=torch.bool, device=population.objs.device)
        first_position = torch.arange(0, n, 2, device=population.objs.device)
        second_position = torch.arange(1, n, 2, device=population.objs.device)
        pop_cv = _constraint_violation(population)
        archive_cv = _constraint_violation(archive)
        choose_population[first_position] = (
            pop_cv[population_index[first_position]] < archive_cv[archive_index[first_position]]
        )
        choose_population[second_position] = (
            angle_population[population_index[second_position]]
            < angle_archive[archive_index[second_position]]
        )

        pop_parent = subset_population(population, population_index)
        archive_parent = subset_population(archive, archive_index)
        return Population(
            torch.where(choose_population.unsqueeze(1), pop_parent.decs, archive_parent.decs),
            torch.where(choose_population.unsqueeze(1), pop_parent.objs, archive_parent.objs),
            torch.where(choose_population.unsqueeze(1), pop_parent.cons, archive_parent.cons),
        )

    @staticmethod
    def _environmental_selection(population: Population, n: int) -> Population:
        """Select the next main population and complementary archive."""
        front_no, max_front = nd_sort(population.objs, population.cons, n_sort=n)
        selected = front_no < max_front
        last = torch.where(front_no == max_front)[0]
        delete_count = int(selected.sum().item()) + last.numel() - n
        if delete_count == 0:
            selected[last] = True
        else:
            zmin = population.objs.min(dim=0).values
            normalized = (population.objs - zmin) / (
                population.objs.max(dim=0).values - zmin + 1e-10
            ) + 1e-10
            deleted = _distance_truncation(normalized[last], delete_count)
            selected[last[~deleted]] = True
        return subset_population(population, torch.where(selected)[0])

    def _solve(self, problem: Problem, output, should_stop, should_pause) -> None:
        """Execute the vectorized evolutionary loop."""
        torch.set_default_dtype(problem.dtype)
        population = problem.initialization()
        archive = Population(
            torch.empty((0, problem.D), dtype=problem.dtype, device=problem.device),
            torch.empty((0, problem.M), dtype=problem.dtype, device=problem.device),
            torch.empty((0, population.cons.shape[1]), dtype=problem.dtype, device=problem.device),
        )

        while self.not_terminated(problem, population, output, should_stop, should_pause):
            all_population = merge_pop(population, archive) if archive.objs.shape[0] else population
            mating = self._mating_selection(population, archive, problem.N)
            offspring = problem.evaluation(operator_ga(problem, mating.decs[: problem.N]))
            archive = self._update_archive(merge_pop(all_population, offspring), problem.N)
            population = self._environmental_selection(merge_pop(population, offspring), problem.N)
