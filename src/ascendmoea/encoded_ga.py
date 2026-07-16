"""Implementation utilities for encoded_ga."""

from __future__ import annotations

import torch

from .core import matlab_round


class _RealProxy:
    """Expose a real-only problem contract for a selected variable subset."""

    def __init__(self, problem, selected: torch.Tensor):
        """Initialize the instance."""
        self.device = problem.device
        self.dtype = problem.dtype
        self.D = problem.encoding_counts[0] + problem.encoding_counts[1]
        self.lower = problem.lower[selected]
        self.upper = problem.upper[selected]
        self.encoding = torch.ones(self.D, dtype=torch.long, device=problem.device)
        real = torch.ones(self.D, dtype=torch.bool, device=problem.device)
        empty = torch.zeros(self.D, dtype=torch.bool, device=problem.device)
        self.encoding_masks = (real, empty, empty, empty, empty)
        self.encoding_counts = (self.D, 0, 0, 0, 0)
        self.real_or_integer_mask = real


def _uniform_crossover(
    parent1: torch.Tensor,
    parent2: torch.Tensor,
    pro_c: float,
    half: bool,
) -> torch.Tensor:
    """Compute uniform crossover."""
    n, d = parent1.shape
    swap = torch.rand((n, d), dtype=parent1.dtype, device=parent1.device) < 0.5
    swap &= torch.rand((n, 1), dtype=parent1.dtype, device=parent1.device) <= pro_c
    child1 = torch.where(swap, parent2, parent1)
    if half:
        return child1
    child2 = torch.where(swap, parent1, parent2)
    return torch.cat([child1, child2], dim=0)


def _label_operator(
    problem,
    parent1: torch.Tensor,
    parent2: torch.Tensor,
    selected: torch.Tensor,
    pro_c: float,
    pro_m: float,
    half: bool,
) -> torch.Tensor:
    """Compute label operator."""
    offspring = _uniform_crossover(parent1, parent2, pro_c, half)
    n, d = offspring.shape
    site = torch.rand_like(offspring) < pro_m / max(d, 1)
    if half:
        random_value = torch.randint(1, d + 1, (n, d), device=offspring.device).to(offspring.dtype)
    else:
        lower = problem.lower[selected]
        upper = problem.upper[selected]
        random_value = matlab_round(
            torch.rand_like(offspring) * (upper - lower).unsqueeze(0) + lower.unsqueeze(0)
        )
    offspring = torch.where(site, random_value, offspring)
    if not half:
        return offspring

    repaired = torch.zeros_like(offspring)
    for row in range(n):
        next_label = 1
        for column in range(d):
            if bool(repaired[row, column] == 0):
                repaired[row, offspring[row] == offspring[row, column]] = next_label
                next_label += 1
    return repaired


def _binary_operator(
    parent1: torch.Tensor,
    parent2: torch.Tensor,
    pro_c: float,
    pro_m: float,
    half: bool,
) -> torch.Tensor:
    """Compute binary operator."""
    offspring = _uniform_crossover(parent1, parent2, pro_c, half)
    site = torch.rand_like(offspring) < pro_m / max(offspring.shape[1], 1)
    return torch.where(site, 1 - offspring, offspring)


def _permutation_operator(
    parent1: torch.Tensor,
    parent2: torch.Tensor,
    pro_c: float,
    half: bool,
) -> torch.Tensor:
    """Compute permutation operator."""
    n, d = parent1.shape
    if half:
        offspring = parent1.clone()
        mate = parent2
        pair_index = torch.arange(n, device=parent1.device)
    else:
        offspring = torch.cat([parent1, parent2], dim=0).clone()
        mate = torch.cat([parent2, parent1], dim=0)
        pair_index = torch.cat([torch.arange(n, device=parent1.device)] * 2, dim=0)

    sampled_cut = torch.randint(0, d, (offspring.shape[0],), device=parent1.device)
    cut = sampled_cut if half else sampled_cut[pair_index]
    crossover_pair = torch.rand((n,), dtype=parent1.dtype, device=parent1.device) < pro_c
    for row in range(offspring.shape[0]):
        if bool(crossover_pair[pair_index[row]]):
            position = int(cut[row].item())
            prefix = offspring[row, : position + 1]
            tail = mate[row, ~torch.isin(mate[row], prefix)]
            offspring[row, position + 1 :] = tail

    moved = torch.randint(0, d, (offspring.shape[0],), device=parent1.device)
    inserted = torch.randint(0, d, (offspring.shape[0],), device=parent1.device)
    for row in range(offspring.shape[0]):
        source = int(moved[row].item())
        target = int(inserted[row].item())
        if target < source:
            order = [*range(target), source, *range(target, source), *range(source + 1, d)]
            offspring[row] = offspring[row, order]
        elif target > source:
            order = [*range(source), *range(source + 1, target), source, *range(target, d)]
            offspring[row] = offspring[row, order]
    return offspring


def operator_ga_encoded(
    problem,
    parent_dec: torch.Tensor,
    pro_c: float = 1.0,
    dis_c: float = 20.0,
    pro_m: float = 1.0,
    dis_m: float = 20.0,
    half: bool = False,
) -> torch.Tensor:
    """Apply the genetic operator independently to each encoding group.

    Args:
        problem: Problem providing bounds and per-variable encoding codes.
        parent_dec: Parent decision tensor arranged in two equal halves.
        pro_c: Crossover probability.
        dis_c: Real-valued crossover distribution index.
        pro_m: Mutation-rate multiplier.
        dis_m: Real-valued mutation distribution index.
        half: Return one child per parent pair when true.

    Returns:
        Encoding-valid offspring decisions.
    """
    from .utils import operator_ga

    pair_count = parent_dec.shape[0] // 2
    parent1 = parent_dec[:pair_count]
    parent2 = parent_dec[pair_count : pair_count * 2]
    offspring_count = pair_count if half else pair_count * 2
    offspring = torch.zeros(
        (offspring_count, parent_dec.shape[1]),
        dtype=parent_dec.dtype,
        device=parent_dec.device,
    )
    total = max(parent_dec.shape[1], 1)
    real_or_integer = problem.real_or_integer_mask
    label = problem.encoding_masks[2]
    binary = problem.encoding_masks[3]
    permutation = problem.encoding_masks[4]

    real_or_integer_count = problem.encoding_counts[0] + problem.encoding_counts[1]
    if real_or_integer_count:
        mutation = pro_m if half else pro_m * real_or_integer_count / total
        proxy = _RealProxy(problem, real_or_integer)
        parents = torch.cat([parent1[:, real_or_integer], parent2[:, real_or_integer]], dim=0)
        offspring[:, real_or_integer] = operator_ga(
            proxy, parents, pro_c, dis_c, mutation, dis_m, half
        )
    if problem.encoding_counts[2]:
        mutation = pro_m if half else pro_m * problem.encoding_counts[2] / total
        offspring[:, label] = _label_operator(
            problem,
            parent1[:, label],
            parent2[:, label],
            label,
            pro_c,
            mutation,
            half,
        )
    if problem.encoding_counts[3]:
        mutation = pro_m if half else pro_m * problem.encoding_counts[3] / total
        offspring[:, binary] = _binary_operator(
            parent1[:, binary], parent2[:, binary], pro_c, mutation, half
        )
    if problem.encoding_counts[4]:
        offspring[:, permutation] = _permutation_operator(
            parent1[:, permutation], parent2[:, permutation], pro_c, half
        )
    return offspring
