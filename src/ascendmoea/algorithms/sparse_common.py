"""Implementation helpers for the sparse_common algorithm."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from ..core import Population, Problem
from ..utils import crowding_distance, merge_pop, nd_sort, tournament_selection


@dataclass
class SparsePopulation:
    """Implement the SparsePopulation multi-objective evolutionary algorithm."""

    population: Population
    dec: torch.Tensor
    mask: torch.Tensor
    front_no: torch.Tensor
    crowd_dis: torch.Tensor


def subset_population(population: Population, index: torch.Tensor) -> Population:
    """Compute subset population."""
    return Population(
        population.decs[index],
        population.objs[index],
        population.cons[index],
    )


def lexicographic_order(rows: torch.Tensor) -> torch.Tensor:
    """Compute lexicographic order."""
    order = torch.arange(rows.shape[0], device=rows.device)
    for column in range(rows.shape[1] - 1, -1, -1):
        local = torch.argsort(rows[order, column], stable=True)
        order = order[local]
    return order


def unique_row_indices(rows: torch.Tensor) -> torch.Tensor:
    """Return MATLAB unique(A,'rows') style sorted first-occurrence indices."""
    if rows.shape[0] <= 1:
        return torch.arange(rows.shape[0], device=rows.device)
    order = lexicographic_order(rows)
    sorted_rows = rows[order]
    first = torch.ones(rows.shape[0], dtype=torch.bool, device=rows.device)
    first[1:] = torch.any(sorted_rows[1:] != sorted_rows[:-1], dim=1)
    return order[first]


def environmental_selection(
    population: Population,
    dec: torch.Tensor,
    mask: torch.Tensor,
    n: int,
    *,
    fallback_to_dec: bool = False,
) -> SparsePopulation:
    """Compute environmental selection."""
    unique = unique_row_indices(population.objs)
    if fallback_to_dec and unique.numel() == 1:
        unique = unique_row_indices(population.decs)
    population = subset_population(population, unique)
    dec = dec[unique]
    mask = mask[unique]
    n = min(int(n), population.objs.shape[0])

    front_no, max_front = nd_sort(population.objs, population.cons, n_sort=n)
    crowd_dis = crowding_distance(population.objs, front_no)
    selected = front_no < max_front
    remaining = n - int(selected.sum().item())
    if remaining > 0:
        last = torch.where(front_no == max_front)[0]
        rank = torch.argsort(crowd_dis[last], descending=True, stable=True)
        selected[last[rank[:remaining]]] = True

    index = torch.where(selected)[0]
    return SparsePopulation(
        subset_population(population, index),
        dec[index],
        mask[index],
        front_no[index],
        crowd_dis[index],
    )


def variable_fitness(
    problem: Problem,
    *,
    keep_all_samples: bool,
) -> tuple[torch.Tensor, Population, torch.Tensor, torch.Tensor]:
    """Compute variable fitness."""
    real = bool(torch.any(problem.encoding != 4).item())
    repeats = 1 + 4 * int(real)
    fitness = torch.zeros(problem.D, dtype=problem.dtype, device=problem.device)
    eye = torch.eye(problem.D, dtype=torch.bool, device=problem.device)
    sampled_populations: list[Population] = []
    sampled_decisions: list[torch.Tensor] = []
    sampled_masks: list[torch.Tensor] = []

    for _ in range(repeats):
        dec = (
            torch.rand((problem.D, problem.D), dtype=problem.dtype, device=problem.device)
            * (problem.upper - problem.lower)
            + problem.lower
        )
        dec[:, problem.encoding == 4] = 1
        population = problem.evaluation(dec * eye.to(dec.dtype))
        augmented_obj = torch.cat([population.objs, population.cons], dim=1)
        front_no, _ = nd_sort(augmented_obj, n_sort=float("inf"))
        fitness += front_no
        if keep_all_samples:
            sampled_populations.append(population)
            sampled_decisions.append(dec)
            sampled_masks.append(eye)
        else:
            sampled_populations = [population]
            sampled_decisions = [dec]
            sampled_masks = [eye]

    return (
        fitness,
        merge_pop(*sampled_populations),
        torch.cat(sampled_decisions, dim=0),
        torch.cat(sampled_masks, dim=0),
    )


def random_decisions(problem: Problem, n: int) -> torch.Tensor:
    """Compute random decisions."""
    dec = torch.rand((n, problem.D), dtype=problem.dtype, device=problem.device)
    dec = dec * (problem.upper - problem.lower) + problem.lower
    dec[:, problem.encoding == 4] = 1
    return dec


def initial_mask(n: int, d: int, fitness: torch.Tensor) -> torch.Tensor:
    # Sampling uses replacement before the selected mask bits are enabled.
    """Compute initial mask."""
    selected_count = torch.ceil(
        torch.rand((n, 1), dtype=fitness.dtype, device=fitness.device) * d
    ).to(torch.long)
    selected_count.clamp_(min=1, max=d)
    winners = tournament_selection(2, n * d, fitness).reshape(n, d)
    active = torch.arange(d, device=fitness.device).view(1, d) < selected_count
    mask = torch.zeros((n, d), dtype=torch.bool, device=fitness.device)
    mask.scatter_(1, winners, active)
    return mask


def _candidate_tournament(
    candidates: torch.Tensor,
    score: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Select one candidate per row by binary tournament; ties use draw one."""
    n, d = candidates.shape
    count = candidates.sum(dim=1)
    valid = count > 0
    draw = torch.rand((n, 2), dtype=score.dtype, device=score.device)
    rank = torch.floor(draw * count.clamp(min=1).unsqueeze(1)).to(torch.long)
    cumulative = candidates.to(torch.long).cumsum(dim=1)
    sampled = torch.argmax((cumulative.unsqueeze(1) > rank.unsqueeze(2)).to(torch.long), dim=2)
    values = torch.gather(score, 1, sampled)
    take_second = values[:, 1] < values[:, 0]
    selected = torch.where(take_second, sampled[:, 1], sampled[:, 0])
    return selected.clamp(max=max(d - 1, 0)), valid


def _update_mask(
    mask: torch.Tensor,
    index: torch.Tensor,
    rows: torch.Tensor,
    value: torch.Tensor,
) -> torch.Tensor:
    """Compute update mask."""
    one_hot = torch.zeros_like(mask)
    one_hot.scatter_(1, index.unsqueeze(1), rows.unsqueeze(1))
    set_one = one_hot & value.unsqueeze(1)
    set_zero = one_hot & ~value.unsqueeze(1)
    return (mask | set_one) & ~set_zero


def sparse_mask_crossover(
    parent1: torch.Tensor,
    parent2: torch.Tensor,
    fitness: torch.Tensor,
) -> torch.Tensor:
    """Compute sparse mask crossover."""
    n, d = parent1.shape
    remove = torch.rand((n,), dtype=fitness.dtype, device=fitness.device) < 0.5
    candidates = torch.where(remove.unsqueeze(1), parent1 & ~parent2, ~parent1 & parent2)
    score = torch.where(
        remove.unsqueeze(1),
        -fitness.view(1, d).expand(n, d),
        fitness.view(1, d).expand(n, d),
    )
    index, valid = _candidate_tournament(candidates, score)
    return _update_mask(parent1.clone(), index, valid, ~remove)


def sparse_mask_mutation(
    mask: torch.Tensor,
    fitness: torch.Tensor,
    allowed: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute sparse mask mutation."""
    n, d = mask.shape
    remove = torch.rand((n,), dtype=fitness.dtype, device=fitness.device) < 0.5
    allowed_mask = torch.ones_like(mask) if allowed is None else allowed
    candidates = torch.where(remove.unsqueeze(1), mask, ~mask) & allowed_mask
    score = torch.where(
        remove.unsqueeze(1),
        -fitness.view(1, d).expand(n, d),
        fitness.view(1, d).expand(n, d),
    )
    index, valid = _candidate_tournament(candidates, score)
    return _update_mask(mask, index, valid, ~remove)


def sbx_first_child(parent1: torch.Tensor, parent2: torch.Tensor) -> torch.Tensor:
    """Compute sbx first child."""
    n, d = parent1.shape
    mu = torch.rand((n, d), dtype=parent1.dtype, device=parent1.device)
    beta = torch.where(
        mu <= 0.5,
        (2 * mu) ** (1 / 21),
        (2 - 2 * mu) ** (-1 / 21),
    )
    sign = torch.where(
        torch.randint(0, 2, (n, d), device=parent1.device) == 0,
        torch.ones_like(beta),
        -torch.ones_like(beta),
    )
    beta *= sign
    beta = torch.where(
        torch.rand((n, d), dtype=parent1.dtype, device=parent1.device) < 0.5,
        torch.ones_like(beta),
        beta,
    )
    return (parent1 + parent2) / 2 + beta * (parent1 - parent2) / 2


def polynomial_mutation_at_sites(
    offspring: torch.Tensor,
    lower: torch.Tensor,
    upper: torch.Tensor,
    sites: torch.Tensor,
    mu: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute polynomial mutation at sites."""
    n, d = offspring.shape
    lower_matrix = lower.view(1, d).expand(n, d)
    upper_matrix = upper.view(1, d).expand(n, d)
    offspring = torch.minimum(torch.maximum(offspring, lower_matrix), upper_matrix)
    if mu is None:
        mu = torch.rand((n, d), dtype=offspring.dtype, device=offspring.device)
    elif mu.shape != offspring.shape:
        mu = mu.expand_as(offspring)

    span = upper_matrix - lower_matrix
    safe_span = torch.where(span != 0, span, torch.ones_like(span))
    left = 1 - (offspring - lower_matrix) / safe_span
    right = 1 - (upper_matrix - offspring) / safe_span
    lower_delta = (2 * mu + (1 - 2 * mu) * left**21) ** (1 / 21) - 1
    upper_delta = 1 - (2 * (1 - mu) + 2 * (mu - 0.5) * right**21) ** (1 / 21)
    delta = torch.where(mu <= 0.5, lower_delta, upper_delta)
    mutated = offspring + span * delta
    return torch.where(sites, mutated, offspring)


def convergence(pop_obj: torch.Tensor) -> torch.Tensor:
    """Compute convergence."""
    front_no, _ = nd_sort(pop_obj, n_sort=float("inf"))
    objective_sum = pop_obj.sum(dim=1)
    spread = objective_sum.max() - objective_sum.min()
    return front_no * spread + objective_sum
