"""Tensorized selection, variation, sampling, and distance operators."""

from __future__ import annotations

import math
import os
from itertools import combinations

import torch

from .core import Population, round_half_away_from_zero, to_tensor
from .device import real_dtype_for_device, select_device

_UNIFORM_POINT_CACHE: dict[tuple[int, int, torch.dtype], torch.Tensor] = {}
_PAIRWISE_DIRECT_ELEMENT_LIMIT = 8_000_000
_COMPILED_TRUNCATION_MIN_SIZE = 512
_COMPILED_TRUNCATION_CALLS = 0
_REFERENCE_TRUNCATION_CALLS = 0
_COMPILED_TRUNCATION_LAST_ERROR: str | None = None


def domination_matrix(pop_obj: torch.Tensor) -> torch.Tensor:
    """Compute pairwise Pareto dominance.

    Returns:
        A Boolean matrix where element ``(i, j)`` is true when row ``i``
        dominates row ``j`` under minimization.
    """
    if pop_obj.ndim != 2 or pop_obj.shape[1] == 0:
        raise ValueError("pop_obj must have shape (N, M) with M >= 1")
    less_equal = pop_obj[:, None, :] <= pop_obj[None, :, :]
    less = pop_obj[:, None, :] < pop_obj[None, :, :]
    return torch.all(less_equal, dim=2) & torch.any(less, dim=2)


def domination_count(pop_obj: torch.Tensor) -> torch.Tensor:
    """Count how many individuals dominate each objective vector."""
    return torch.sum(domination_matrix(pop_obj), dim=0)


def nd_sort(
    pop_obj: torch.Tensor, pop_con: torch.Tensor | None = None, n_sort: int | float = float("inf")
):
    """Assign constrained nondominated front numbers.

    Args:
        pop_obj: Objective matrix with shape ``(N, M)``.
        pop_con: Optional inequality-constraint matrix where positive values
            represent violations.
        n_sort: Minimum number of individuals that must receive finite fronts.

    Returns:
        A pair containing each individual's front number and the last assigned
        front number.
    """
    if pop_obj.ndim != 2 or pop_obj.shape[1] == 0:
        raise ValueError("pop_obj must have shape (N, M) with M >= 1")
    if pop_con is not None and (pop_con.ndim != 2 or pop_con.shape[0] != pop_obj.shape[0]):
        raise ValueError("pop_con must be a matrix with the same row count as pop_obj")
    if n_sort < 0:
        raise ValueError("n_sort must be nonnegative")
    obj = pop_obj.clone()
    n = obj.shape[0]
    if pop_con is not None:
        infeasible = torch.any(pop_con > 0, dim=1)
        if torch.any(infeasible):
            cv = torch.sum(torch.clamp(pop_con[infeasible], min=0.0), dim=1, keepdim=True)
            obj[infeasible] = torch.max(obj, dim=0, keepdim=True).values + cv

    dom_matrix = domination_matrix(obj)
    front_no = torch.full((n,), float("inf"), dtype=obj.dtype, device=obj.device)
    dominated_count = torch.sum(dom_matrix, dim=0)
    current_front_idx = 1
    sorted_count = 0

    while sorted_count < min(n_sort, n):
        current_front = torch.where((dominated_count == 0) & torch.isinf(front_no))[0]
        if current_front.numel() == 0:
            break
        front_no[current_front] = float(current_front_idx)
        sorted_count += int(current_front.numel())
        dominated_count = dominated_count - torch.sum(dom_matrix[current_front], dim=0)
        current_front_idx += 1

    finite = front_no[torch.isfinite(front_no)]
    max_f = int(torch.max(finite).item()) if finite.numel() else 0
    return front_no, max_f


def tournament_selection(k: int, n: int, *fitness: torch.Tensor) -> torch.Tensor:
    """Select parent indices by lexicographic ``k``-way tournaments.

    Fitness criteria are minimized and applied from left to right. Stable
    ordering preserves deterministic tie semantics for a fixed random seed.
    """
    if k < 1 or n < 0 or not fitness:
        raise ValueError("tournament_selection requires k >= 1, n >= 0, and fitness")
    fit = torch.column_stack([f.reshape(-1) for f in fitness])
    if fit.shape[0] == 0:
        raise ValueError("fitness arrays must not be empty")
    if any(f.numel() != fit.shape[0] for f in fitness):
        raise ValueError("all fitness arrays must have the same number of elements")
    parents = torch.randint(0, fit.shape[0], (k, n), device=fit.device)
    order = torch.arange(k, device=fit.device).view(k, 1).expand(k, n)
    for criterion in range(fit.shape[1] - 1, -1, -1):
        values = fit[parents, criterion]
        ranked_values = torch.gather(values, 0, order)
        local_order = torch.argsort(ranked_values, dim=0, stable=True)
        order = torch.gather(order, 0, local_order)
    best = order[0]
    return parents[best, torch.arange(n, device=fit.device)]


def crowding_distance(pop_obj: torch.Tensor, front_no: torch.Tensor) -> torch.Tensor:
    """Calculate objective-space crowding distance within each front.

    Boundary points receive infinity. The returned vector has one value per
    row of ``pop_obj`` and remains on the same device.
    """
    if pop_obj.ndim != 2 or pop_obj.shape[1] == 0:
        raise ValueError("pop_obj must have shape (N, M) with M >= 1")
    if front_no.ndim != 1 or front_no.numel() != pop_obj.shape[0]:
        raise ValueError("front_no must contain one value per objective row")
    n, m = pop_obj.shape
    cd = torch.zeros(n, dtype=pop_obj.dtype, device=pop_obj.device)
    fronts = torch.unique(front_no[torch.isfinite(front_no)]).tolist()

    for f in fronts:
        idx = torch.where(front_no == f)[0]
        if idx.numel() == 0:
            continue
        fvals = pop_obj[idx]
        fmax = torch.max(fvals, dim=0).values
        fmin = torch.min(fvals, dim=0).values
        order_local = torch.argsort(fvals, dim=0, stable=True)
        sorted_vals = torch.gather(fvals, 0, order_local)
        denom = fmax - fmin
        increment = (sorted_vals[2:, :] - sorted_vals[:-2, :]) / denom.unsqueeze(0)
        by_point = torch.zeros_like(fvals)
        by_point.scatter_(0, order_local[1:-1, :], increment)
        boundary = torch.zeros_like(fvals, dtype=torch.bool)
        boundary.scatter_(
            0,
            order_local[[0, -1], :],
            torch.ones((2, m), dtype=torch.bool, device=pop_obj.device),
        )
        objective = torch.arange(m, device=pop_obj.device).unsqueeze(0)
        last_boundary = torch.where(boundary, objective, -1).max(dim=1).values
        has_boundary = last_boundary >= 0
        nan_after_boundary = (torch.isnan(by_point) & (objective > last_boundary.unsqueeze(1))).any(
            dim=1
        )
        local = torch.where(
            has_boundary,
            torch.full((idx.numel(),), torch.inf, dtype=pop_obj.dtype, device=pop_obj.device),
            by_point.sum(dim=1),
        )
        local = torch.where(
            nan_after_boundary,
            torch.full_like(local, torch.nan),
            local,
        )
        cd[idx] = local
    return cd


def _polynomial_mutation(problem, off: torch.Tensor, pro_m: float, dis_m: float) -> torch.Tensor:
    """Apply bounded polynomial mutation to selected real-valued genes."""
    _, d = off.shape
    device = off.device
    dtype = off.dtype
    lower = problem.lower.unsqueeze(0).expand_as(off)
    upper = problem.upper.unsqueeze(0).expand_as(off)
    off = torch.minimum(torch.maximum(off, lower), upper)
    den = torch.clamp(upper - lower, min=1e-12)

    site = torch.rand(off.shape, device=device, dtype=dtype) < (pro_m / max(d, 1))
    mu = torch.rand(off.shape, device=device, dtype=dtype)
    left = torch.clamp(1 - (off - lower) / den, min=0.0)
    right = torch.clamp(1 - (upper - off) / den, min=0.0)
    delta_lower = (2 * mu + (1 - 2 * mu) * left ** (dis_m + 1)) ** (1 / (dis_m + 1)) - 1
    delta_upper = 1 - (2 * (1 - mu) + 2 * (mu - 0.5) * right ** (dis_m + 1)) ** (1 / (dis_m + 1))
    delta = torch.where(mu <= 0.5, delta_lower, delta_upper)
    off = torch.where(site, off + (upper - lower) * delta, off)
    return torch.minimum(torch.maximum(off, lower), upper)


def _ga_real(
    parent1: torch.Tensor,
    parent2: torch.Tensor,
    lower: torch.Tensor,
    upper: torch.Tensor,
    pro_c: float,
    dis_c: float,
    pro_m: float,
    dis_m: float,
    half: bool,
) -> torch.Tensor:
    """Generate real-coded offspring with SBX and polynomial mutation."""
    n, d = parent1.shape
    mu = torch.rand((n, d), dtype=parent1.dtype, device=parent1.device)
    beta = torch.where(
        mu <= 0.5,
        (2 * mu) ** (1.0 / (dis_c + 1.0)),
        (2 - 2 * mu) ** (-1.0 / (dis_c + 1.0)),
    )
    sign = torch.where(
        torch.randint(0, 2, (n, d), device=parent1.device) == 0,
        torch.ones((), dtype=parent1.dtype, device=parent1.device),
        -torch.ones((), dtype=parent1.dtype, device=parent1.device),
    )
    beta = beta * sign
    beta = torch.where(
        torch.rand((n, d), dtype=parent1.dtype, device=parent1.device) < 0.5,
        torch.ones_like(beta),
        beta,
    )
    beta = torch.where(
        torch.rand((n, 1), dtype=parent1.dtype, device=parent1.device) > pro_c,
        torch.ones_like(beta),
        beta,
    )
    child1 = (parent1 + parent2) / 2 + beta * (parent1 - parent2) / 2
    child2 = (parent1 + parent2) / 2 - beta * (parent1 - parent2) / 2
    offspring = child1 if half else torch.cat([child1, child2], dim=0)

    lower = lower.unsqueeze(0).expand_as(offspring)
    upper = upper.unsqueeze(0).expand_as(offspring)
    offspring = torch.minimum(torch.maximum(offspring, lower), upper)
    site = torch.rand_like(offspring) < (pro_m / max(d, 1))
    mutation = torch.rand_like(offspring)
    span = upper - lower
    denominator = torch.clamp(span, min=torch.finfo(offspring.dtype).eps)
    left = torch.clamp(1 - (offspring - lower) / denominator, min=0.0)
    right = torch.clamp(1 - (upper - offspring) / denominator, min=0.0)
    delta_lower = (2 * mutation + (1 - 2 * mutation) * left ** (dis_m + 1)) ** (1 / (dis_m + 1)) - 1
    delta_upper = 1 - (2 * (1 - mutation) + 2 * (mutation - 0.5) * right ** (dis_m + 1)) ** (
        1 / (dis_m + 1)
    )
    delta = torch.where(mutation <= 0.5, delta_lower, delta_upper)
    offspring = torch.where(site & (span > 0), offspring + span * delta, offspring)
    return torch.minimum(torch.maximum(offspring, lower), upper)


def _uniform_crossover(
    parent1: torch.Tensor,
    parent2: torch.Tensor,
    pro_c: float,
    half: bool,
) -> torch.Tensor:
    """Exchange genes between paired parents with Bernoulli masks."""
    n, d = parent1.shape
    swap = torch.rand((n, d), dtype=parent1.dtype, device=parent1.device) < 0.5
    swap &= torch.rand((n, 1), dtype=parent1.dtype, device=parent1.device) <= pro_c
    child1 = torch.where(swap, parent2, parent1)
    if half:
        return child1
    child2 = torch.where(swap, parent1, parent2)
    return torch.cat([child1, child2], dim=0)


def _ga_label(
    parent1: torch.Tensor,
    parent2: torch.Tensor,
    lower: torch.Tensor,
    upper: torch.Tensor,
    pro_c: float,
    pro_m: float,
    half: bool,
) -> torch.Tensor:
    """Generate offspring for categorical label variables."""
    offspring = _uniform_crossover(parent1, parent2, pro_c, half)
    n, d = offspring.shape
    site = torch.rand((n, d), dtype=offspring.dtype, device=offspring.device) < (pro_m / max(d, 1))
    if half:
        random_label = torch.randint(1, d + 1, (n, d), device=offspring.device).to(offspring.dtype)
    else:
        random_label = round_half_away_from_zero(
            torch.rand((n, d), dtype=offspring.dtype, device=offspring.device)
            * (upper - lower).unsqueeze(0)
            + lower.unsqueeze(0)
        )
    offspring = torch.where(site, random_label, offspring)
    if not half:
        return offspring

    positions = torch.arange(d, device=offspring.device)
    equal = offspring.unsqueeze(2) == offspring.unsqueeze(1)
    first = (
        torch.where(
            equal,
            positions.view(1, 1, d),
            torch.full((), d, dtype=positions.dtype, device=positions.device),
        )
        .min(dim=2)
        .values
    )
    first_position = first == positions.unsqueeze(0)
    rank_at_position = torch.cumsum(first_position.to(torch.long), dim=1)
    return rank_at_position.gather(1, first).to(offspring.dtype)


def _ga_binary(
    parent1: torch.Tensor,
    parent2: torch.Tensor,
    pro_c: float,
    pro_m: float,
    half: bool,
) -> torch.Tensor:
    """Generate offspring for binary variables by crossover and bit mutation."""
    offspring = _uniform_crossover(parent1, parent2, pro_c, half)
    site = torch.rand_like(offspring) < (pro_m / max(offspring.shape[1], 1))
    return torch.where(site, 1 - offspring, offspring)


def _order_crossover(
    base: torch.Tensor,
    mate: torch.Tensor,
    cut: torch.Tensor,
    crossover: torch.Tensor,
) -> torch.Tensor:
    """Build permutation children while preserving relative element order."""
    n, d = base.shape
    positions = torch.arange(d, device=base.device).unsqueeze(0).expand(n, d)
    prefix = positions <= cut.unsqueeze(1)
    used = ((mate.unsqueeze(2) == base.unsqueeze(1)) & prefix.unsqueeze(1)).any(dim=2)
    keep = ~used
    compact_order = torch.argsort(torch.where(keep, positions, positions + d), dim=1, stable=True)
    compact = mate.gather(1, compact_order)
    tail_index = torch.clamp(positions - cut.unsqueeze(1) - 1, min=0, max=d - 1)
    crossed = torch.where(prefix, base, compact.gather(1, tail_index))
    return torch.where(crossover.unsqueeze(1), crossed, base)


def _slight_permutation_mutation(offspring: torch.Tensor) -> torch.Tensor:
    """Perturb permutations with swap, insertion, or reversal moves."""
    n, d = offspring.shape
    if d <= 1:
        return offspring
    moved = torch.randint(0, d, (n,), device=offspring.device)
    inserted = torch.randint(0, d, (n,), device=offspring.device)
    positions = torch.arange(d, device=offspring.device).unsqueeze(0).expand(n, d)
    source = positions.clone()
    move_left = inserted < moved
    source = torch.where(
        move_left.unsqueeze(1) & (positions == inserted.unsqueeze(1)),
        moved.unsqueeze(1),
        source,
    )
    source = torch.where(
        move_left.unsqueeze(1)
        & (positions > inserted.unsqueeze(1))
        & (positions <= moved.unsqueeze(1)),
        positions - 1,
        source,
    )
    move_right = inserted > moved
    source = torch.where(
        move_right.unsqueeze(1)
        & (positions >= moved.unsqueeze(1))
        & (positions < (inserted - 1).unsqueeze(1)),
        positions + 1,
        source,
    )
    source = torch.where(
        move_right.unsqueeze(1) & (positions == (inserted - 1).unsqueeze(1)),
        moved.unsqueeze(1),
        source,
    )
    return offspring.gather(1, source)


def _ga_permutation(
    parent1: torch.Tensor,
    parent2: torch.Tensor,
    pro_c: float,
    half: bool,
) -> torch.Tensor:
    """Generate permutation offspring with order crossover and mutation."""
    n, d = parent1.shape
    if half:
        base = parent1
        mate = parent2
        crossover = torch.rand((n,), dtype=parent1.dtype, device=parent1.device) < pro_c
    else:
        base = torch.cat([parent1, parent2], dim=0)
        mate = torch.cat([parent2, parent1], dim=0)
        pair_crossover = torch.rand((n,), dtype=parent1.dtype, device=parent1.device) < pro_c
        crossover = torch.cat([pair_crossover, pair_crossover], dim=0)
    sampled_cut = torch.randint(0, d, (base.shape[0],), device=base.device)
    cut = sampled_cut if half else torch.cat([sampled_cut[:n], sampled_cut[:n]])
    return _slight_permutation_mutation(_order_crossover(base, mate, cut, crossover))


def operator_ga(
    problem, parent_dec: torch.Tensor, pro_c=1.0, dis_c=20.0, pro_m=1.0, dis_m=20.0, half=False
):
    """Apply encoding-aware crossover and mutation to paired parents.

    Args:
        problem: Problem providing bounds and variable encodings.
        parent_dec: Parent decisions; the first and second halves form pairs.
        pro_c: Crossover probability.
        dis_c: Simulated-binary crossover distribution index.
        pro_m: Expected mutation rate multiplier.
        dis_m: Polynomial mutation distribution index.
        half: Return one child per pair instead of two.

    Returns:
        A tensor of offspring decisions on the problem device.
    """
    parent_dec = to_tensor(parent_dec, device=problem.device)
    if parent_dec.ndim != 2 or parent_dec.shape[1] != problem.D:
        raise ValueError(f"parent_dec must have shape (N, D={problem.D})")
    if parent_dec.shape[0] < 2:
        raise ValueError("operator_ga requires at least two parent rows")
    pair_count = parent_dec.shape[0] // 2
    parent1 = parent_dec[:pair_count]
    parent2 = parent_dec[pair_count : pair_count * 2]
    offspring_count = pair_count if half else pair_count * 2
    offspring = torch.zeros(
        (offspring_count, parent_dec.shape[1]),
        dtype=parent_dec.dtype,
        device=parent_dec.device,
    )
    real_or_integer = problem.real_or_integer_mask
    label = problem.encoding_masks[2]
    binary = problem.encoding_masks[3]
    permutation = problem.encoding_masks[4]
    total_variables = max(parent_dec.shape[1], 1)

    real_or_integer_count = problem.encoding_counts[0] + problem.encoding_counts[1]
    if real_or_integer_count:
        mutation = pro_m if half else pro_m * real_or_integer_count / total_variables
        offspring[:, real_or_integer] = _ga_real(
            parent1[:, real_or_integer],
            parent2[:, real_or_integer],
            problem.lower[real_or_integer],
            problem.upper[real_or_integer],
            pro_c,
            dis_c,
            mutation,
            dis_m,
            half,
        )
    if problem.encoding_counts[2]:
        mutation = pro_m if half else pro_m * problem.encoding_counts[2] / total_variables
        offspring[:, label] = _ga_label(
            parent1[:, label],
            parent2[:, label],
            problem.lower[label],
            problem.upper[label],
            pro_c,
            mutation,
            half,
        )
    if problem.encoding_counts[3]:
        mutation = pro_m if half else pro_m * problem.encoding_counts[3] / total_variables
        offspring[:, binary] = _ga_binary(
            parent1[:, binary], parent2[:, binary], pro_c, mutation, half
        )
    if problem.encoding_counts[4]:
        offspring[:, permutation] = _ga_permutation(
            parent1[:, permutation], parent2[:, permutation], pro_c, half
        )
    return offspring


def operator_de(
    problem,
    p1: torch.Tensor,
    p2: torch.Tensor,
    p3: torch.Tensor,
    cr=1.0,
    f=0.5,
    pro_m=1.0,
    dis_m=20.0,
):
    """Apply differential evolution followed by polynomial mutation.

    The mutant is ``p1 + f * (p2 - p3)`` and each variable is accepted with
    crossover probability ``cr`` before bounded mutation.
    """
    p1 = to_tensor(p1, device=problem.device)
    p2 = to_tensor(p2, device=problem.device)
    p3 = to_tensor(p3, device=problem.device)
    if p1.ndim != 2 or p1.shape[1] != problem.D:
        raise ValueError(f"p1 must have shape (N, D={problem.D})")
    if p2.shape != p1.shape or p3.shape != p1.shape:
        raise ValueError("p1, p2, and p3 must have identical shapes")
    n, d = p1.shape
    device = p1.device
    dtype = p1.dtype

    site = torch.rand((n, d), device=device, dtype=dtype) < cr
    mutant = p1 + f * (p2 - p3)
    off = torch.where(site, mutant, p1)
    return _polynomial_mutation(problem, off, pro_m, dis_m)


def roulette_selection(n: int, fitness: torch.Tensor) -> torch.Tensor:
    """Draw ``n`` indices with probabilities inverse to minimized fitness."""
    if n < 0:
        raise ValueError("n must be nonnegative")
    fit = fitness.reshape(-1)
    if fit.numel() == 0 or not torch.all(torch.isfinite(fit)):
        raise ValueError("fitness must be nonempty and finite")
    if not fit.is_floating_point():
        fit = fit.to(torch.get_default_dtype())
    fit = (
        fit
        - torch.minimum(torch.min(fit), torch.zeros((), dtype=fit.dtype, device=fit.device))
        + 1e-6
    )
    cdf = torch.cumsum(1 / fit, dim=0)
    cdf = cdf / cdf[-1]
    r = torch.rand((n,), dtype=cdf.dtype, device=cdf.device)
    return torch.clamp(torch.searchsorted(cdf, r, right=False), max=fit.numel() - 1)


def uniform_point(n: int, m: int, device: torch.device = None) -> tuple[torch.Tensor, int]:
    """Generate simplex-lattice reference vectors.

    Args:
        n: Requested approximate number of vectors.
        m: Objective-space dimension.
        device: Destination device; automatic selection is used when omitted.

    Returns:
        The generated reference vectors and their actual count.
    """
    if n < 1 or m < 2:
        raise ValueError("uniform_point requires n >= 1 and m >= 2")
    if device is None:
        device = select_device()
    dtype = real_dtype_for_device(device)
    key = (int(n), int(m), dtype)
    cached = _UNIFORM_POINT_CACHE.get(key, None)
    if cached is not None:
        w = cached.to(device=device)
        return w, w.shape[0]

    h1 = 1
    while math.comb(h1 + m, m - 1) <= n:
        h1 += 1
    comb_list = list(combinations(range(1, h1 + m), m - 1))
    w_raw = torch.tensor(comb_list, dtype=dtype, device=device)
    shift = torch.arange(1, m, dtype=dtype, device=device)
    w = w_raw - shift
    h1_t = torch.full((w.shape[0], 1), h1, dtype=dtype, device=device)
    zeros_t = torch.zeros((w.shape[0], 1), dtype=dtype, device=device)
    w = torch.cat([w, h1_t], dim=1) - torch.cat([zeros_t, w], dim=1)
    w = w / h1
    if h1 < m:
        h2 = 0
        while math.comb(h1 + m - 1, m - 1) + math.comb(h2 + m, m - 1) <= n:
            h2 += 1
        if h2 > 0:
            comb_list2 = list(combinations(range(1, h2 + m), m - 1))
            w2_raw = torch.tensor(comb_list2, dtype=dtype, device=device)
            w2 = w2_raw - shift
            h2_t = torch.full((w2.shape[0], 1), h2, dtype=dtype, device=device)
            zeros2_t = torch.zeros((w2.shape[0], 1), dtype=dtype, device=device)
            w2 = torch.cat([w2, h2_t], dim=1) - torch.cat([zeros2_t, w2], dim=1)
            w2 = w2 / h2
            w = torch.cat([w, w2 / 2.0 + 1.0 / (2.0 * m)], dim=0)
    w = torch.clamp(w, min=1e-6)
    _UNIFORM_POINT_CACHE[key] = w.cpu()
    w = w.to(device=device)
    return w, w.shape[0]


def merge_pop(*pops: Population) -> Population:
    """Concatenate populations along their individual dimension."""
    if len(pops) == 0:
        raise ValueError("merge_pop requires at least one population")
    return Population(
        torch.cat([p.decs for p in pops], dim=0),
        torch.cat([p.objs for p in pops], dim=0),
        torch.cat([p.cons for p in pops], dim=0),
    )


def _pairwise_dist_centered(
    a: torch.Tensor,
    b: torch.Tensor,
    center: torch.Tensor,
) -> torch.Tensor:
    """Compute pairwise distances with a caller-supplied numerical center."""
    centered_a = a - center
    centered_b = b - center
    product = centered_a @ centered_b.transpose(0, 1)
    norm_a = torch.sum(centered_a * centered_a, dim=1, keepdim=True)
    norm_b = torch.sum(centered_b * centered_b, dim=1).unsqueeze(0)
    squared = norm_a + norm_b - 2 * product
    magnitude = norm_a + norm_b + 2 * torch.abs(product)
    threshold = 32 * torch.finfo(a.dtype).eps * torch.clamp(magnitude, min=1)
    uncertain = squared <= threshold
    row, column = torch.where(uncertain)
    if row.numel():
        difference = a[row] - b[column]
        squared[row, column] = torch.sum(difference * difference, dim=1)
    return torch.sqrt(torch.clamp(squared, min=0.0))


def pairwise_dist(a: torch.Tensor, b: torch.Tensor):
    """Compute a numerically corrected Euclidean distance matrix.

    A direct broadcast path handles moderate inputs. Larger inputs use a
    centered matrix-product formulation and selectively recompute distances
    susceptible to cancellation.
    """
    a = to_tensor(a)
    b = to_tensor(b, device=a.device)
    if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[1]:
        raise ValueError("pairwise_dist expects two matrices with equal feature dimensions")
    if a.shape[0] == 0 or b.shape[0] == 0:
        return torch.empty((a.shape[0], b.shape[0]), dtype=a.dtype, device=a.device)
    if a.shape[0] * b.shape[0] * a.shape[1] <= _PAIRWISE_DIRECT_ELEMENT_LIMIT:
        return torch.linalg.vector_norm(a.unsqueeze(1) - b.unsqueeze(0), dim=2)
    center = (a.sum(dim=0) + b.sum(dim=0)) / (a.shape[0] + b.shape[0])
    distance = _pairwise_dist_centered(a, b, center)
    if a.data_ptr() == b.data_ptr() and a.shape[0] == b.shape[0]:
        distance.fill_diagonal_(0.0)
    return distance


def sequential_euclidean_truncation(
    *spaces: torch.Tensor,
    delete_count: int,
) -> torch.Tensor:
    """Apply exact truncation to summed Euclidean distance spaces.

    Every distance space must contain the same individuals and use the same
    device and dtype. The complete pairwise matrices are evaluated before the
    deterministic deletion recurrence.
    """
    if not spaces:
        raise ValueError("at least one distance space is required")
    first = spaces[0]
    if first.ndim != 2:
        raise ValueError("distance spaces must be matrices")
    n = first.shape[0]
    for space in spaces:
        if (
            space.ndim != 2
            or space.shape[0] != n
            or space.device != first.device
            or space.dtype != first.dtype
        ):
            raise ValueError(
                "distance spaces must have equal row counts, devices, and dtypes"
            )
    if delete_count <= 0:
        return torch.zeros(n, dtype=torch.bool, device=first.device)

    distance = pairwise_dist(first, first)
    for space in spaces[1:]:
        distance.add_(pairwise_dist(space, space))
    return sequential_distance_truncation(distance, delete_count)


def sequential_distance_truncation(distance: torch.Tensor, delete_count: int) -> torch.Tensor:
    """Apply deterministic sequential truncation using stable distance orderings."""
    if distance.ndim != 2 or distance.shape[0] != distance.shape[1]:
        raise ValueError("distance must be a square matrix")
    n = distance.shape[0]
    if delete_count <= 0:
        return torch.zeros(n, dtype=torch.bool, device=distance.device)

    output_device = distance.device
    if output_device.type == "npu":
        # The deletion recurrence is sequential and launches only tiny kernels.
        # Keep the O(n^2) distance construction on NPU, then execute this
        # deterministic control section on CPU and return only its mask.
        distance = distance.cpu()

    distance = distance.clone()
    distance.fill_diagonal_(float("inf"))
    sorted_distance, neighbor_order = torch.sort(distance, dim=1, stable=True)

    global _COMPILED_TRUNCATION_CALLS
    global _REFERENCE_TRUNCATION_CALLS
    global _COMPILED_TRUNCATION_LAST_ERROR
    accelerator_mode = os.environ.get("ASCENDMOEA_COMPILED_TRUNCATION", "auto").strip().lower()
    if accelerator_mode not in {"auto", "off", "required"}:
        raise ValueError("ASCENDMOEA_COMPILED_TRUNCATION must be auto, off, or required")
    if accelerator_mode != "off" and n >= _COMPILED_TRUNCATION_MIN_SIZE:
        try:
            from .compiled_truncation import sequential_recurrence

            compiled = sequential_recurrence(
                sorted_distance, neighbor_order, min(int(delete_count), n)
            )
            if compiled is not None:
                _COMPILED_TRUNCATION_CALLS += 1
                _COMPILED_TRUNCATION_LAST_ERROR = None
                deleted = compiled.to(output_device)
                return deleted
            _COMPILED_TRUNCATION_LAST_ERROR = "numba is not installed"
        except Exception as exception:
            _COMPILED_TRUNCATION_LAST_ERROR = repr(exception)
            if accelerator_mode == "required":
                raise RuntimeError("compiled sequential truncation failed") from exception
        if accelerator_mode == "required":
            raise RuntimeError(
                "compiled sequential truncation is required but unavailable: "
                f"{_COMPILED_TRUNCATION_LAST_ERROR}"
            )

    _REFERENCE_TRUNCATION_CALLS += 1
    active = torch.ones(n, dtype=torch.bool, device=distance.device)
    cursor = torch.zeros(n, dtype=torch.long, device=distance.device)
    rows = torch.arange(n, device=distance.device)

    def matlab_minimum_mask(values: torch.Tensor) -> torch.Tensor:
        """Match MATLAB ascending order, where NaNs follow all numeric values."""
        numeric = ~torch.isnan(values)
        if torch.any(numeric):
            minimum = torch.min(values[numeric])
            return numeric & (values == minimum)
        return torch.ones_like(values, dtype=torch.bool)

    for _ in range(min(int(delete_count), n)):
        remain = rows[active]
        nearest = sorted_distance[remain, cursor[remain]]
        candidates = remain[matlab_minimum_mask(nearest)]
        if candidates.numel() > 1:
            alive_neighbors = active[neighbor_order[candidates]]
            sequences = sorted_distance[candidates][alive_neighbors].reshape(
                candidates.numel(), remain.numel()
            )
            tied = torch.arange(candidates.numel(), device=distance.device)
            for column in range(sequences.shape[1]):
                values = sequences[tied, column]
                tied = tied[matlab_minimum_mask(values)]
                if tied.numel() == 1:
                    break
            chosen = candidates[tied[0]]
        else:
            chosen = candidates[0]

        active[chosen] = False
        current_neighbor = neighbor_order[rows, cursor]
        affected = rows[active & (current_neighbor == chosen)]
        if affected.numel() > 0:
            alive_by_distance = active[neighbor_order[affected]]
            cursor[affected] = torch.argmax(alive_by_distance.to(torch.int8), dim=1)
    deleted = ~active
    return deleted.to(output_device) if deleted.device != output_device else deleted


def sequential_truncation_diagnostics() -> dict[str, object]:
    """Return compiled and reference truncation call diagnostics."""
    return {
        "minimum_size": _COMPILED_TRUNCATION_MIN_SIZE,
        "compiled_calls": _COMPILED_TRUNCATION_CALLS,
        "reference_calls": _REFERENCE_TRUNCATION_CALLS,
        "last_error": _COMPILED_TRUNCATION_LAST_ERROR,
    }


def reset_sequential_truncation_diagnostics() -> None:
    """Reset process-local sequential truncation diagnostics."""
    global _COMPILED_TRUNCATION_CALLS
    global _REFERENCE_TRUNCATION_CALLS
    global _COMPILED_TRUNCATION_LAST_ERROR
    _COMPILED_TRUNCATION_CALLS = 0
    _REFERENCE_TRUNCATION_CALLS = 0
    _COMPILED_TRUNCATION_LAST_ERROR = None


def warmup_sequential_truncation_accelerator(
    dtype: torch.dtype | None = None,
) -> dict[str, object]:
    """Compile and validate the optional exact CPU truncation accelerator."""
    if dtype is None:
        dtype = torch.get_default_dtype()
    mode = os.environ.get("ASCENDMOEA_COMPILED_TRUNCATION", "auto").strip().lower()
    if mode not in {"auto", "off", "required"}:
        raise ValueError("ASCENDMOEA_COMPILED_TRUNCATION must be auto, off, or required")
    if mode == "off":
        return {"available": False, "mode": mode, "reason": "disabled"}
    try:
        from .compiled_truncation import warmup

        result = warmup(dtype)
    except Exception as exception:
        if mode == "required":
            raise RuntimeError("compiled truncation warmup failed") from exception
        return {
            "available": False,
            "mode": mode,
            "reason": repr(exception),
        }
    result["mode"] = mode
    if mode == "required" and not result["available"]:
        raise RuntimeError("compiled truncation is required but numba is unavailable")
    return result
