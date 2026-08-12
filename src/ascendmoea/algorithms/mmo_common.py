"""Implementation helpers for the mmo_common algorithm."""

from __future__ import annotations

import math

import torch

from ..core import Population
from ..utils import (
    domination_matrix,
    merge_pop,
    nd_sort,
    pairwise_dist,
    sequential_euclidean_truncation,
)


def _subset(pop: Population, idx: torch.Tensor) -> Population:
    """Select population rows while preserving all population fields."""
    return Population(pop.decs[idx], pop.objs[idx], pop.cons[idx])


def _cat(*pops: Population) -> Population:
    """Concatenate populations along the individual dimension."""
    return merge_pop(*[p for p in pops if p.decs.shape[0] > 0])


def _empty_like(pop: Population) -> Population:
    """Create an empty population with matching feature dimensions."""
    return Population(pop.decs[:0], pop.objs[:0], pop.cons[:0])


def _dominance_strength_fitness(pop_obj: torch.Tensor, local_mask: torch.Tensor | None = None):
    """Calculate strength-based fitness from pairwise dominance."""
    dom = domination_matrix(pop_obj)
    if local_mask is not None:
        dom = dom & local_mask
    s = torch.sum(dom, dim=1).to(pop_obj.dtype)
    r = torch.sum(dom.T.to(pop_obj.dtype) * s.reshape(1, -1), dim=1)
    return dom, r


def _kth_density(x: torch.Tensor, k: int) -> torch.Tensor:
    """Estimate density from each solution's kth-nearest distance."""
    n = x.shape[0]
    if n <= 1:
        return torch.zeros(n, dtype=x.dtype, device=x.device)
    k = max(1, min(k, n - 1))
    dist = pairwise_dist(x, x)
    dist.fill_diagonal_(float("inf"))
    kth = torch.kthvalue(dist, k, dim=1).values
    return 1.0 / (kth + 2.0)


def _harmonic_crowding(dec: torch.Tensor) -> torch.Tensor:
    """Compute harmonic-mean crowding in normalized decision space."""
    n = dec.shape[0]
    if n <= 1:
        return torch.full((n,), float("nan"), dtype=dec.dtype, device=dec.device)
    z_min = torch.min(dec, dim=0).values
    z_max = torch.max(dec, dim=0).values
    span = (z_max - z_min).clamp_min(torch.finfo(dec.dtype).eps)
    norm = (dec - z_min) / span
    dist = pairwise_dist(norm, norm)
    vals = torch.sort(dist, dim=1).values[:, 1:]
    return (n - 1) / torch.sum(1.0 / vals, dim=1)


def _sum_nearest_distance(x: torch.Tensor, k: int = 3) -> torch.Tensor:
    """Sum each solution's nearest-neighbor distances in decision space."""
    n = x.shape[0]
    if n <= 1:
        return torch.ones(n, dtype=x.dtype, device=x.device)
    dist = torch.sort(pairwise_dist(x, x), dim=0).values
    k = min(k, dist.shape[0])
    return torch.sum(dist[:k, :], dim=0)


def _local_convergence(pop: Population) -> torch.Tensor:
    """Count locally dominating neighbors for every individual."""
    n, d = pop.decs.shape
    if n == 0:
        return torch.empty(0, dtype=pop.decs.dtype, device=pop.decs.device)
    dist = pairwise_dist(pop.decs, pop.decs)
    span = torch.max(pop.decs, dim=0).values - torch.min(pop.decs, dim=0).values
    v = 0.2 * torch.prod(span).pow(1.0 / max(d, 1))
    neighbor = dist < v
    neighbor.fill_diagonal_(True)
    dominated_by = domination_matrix(pop.objs).T
    return torch.sum(dominated_by & neighbor, dim=1).to(pop.decs.dtype) / torch.sum(
        neighbor, dim=1
    ).clamp_min(1)


def _lex_sort_indices(*columns: torch.Tensor, descending: bool = False) -> torch.Tensor:
    """Return stable lexicographic row indices for deterministic tie breaking."""
    order = torch.arange(columns[0].numel(), device=columns[0].device)
    for col in reversed(columns):
        order = order[torch.argsort(col[order], descending=descending, stable=True)]
    return order


def _truncation_by_distance(*spaces: torch.Tensor, k: int) -> torch.Tensor:
    """Select removals by iterative lexicographic nearest-distance comparison."""
    return sequential_euclidean_truncation(*spaces, delete_count=k)


def _cmmo_fitness(pop: Population):
    """Calculate CMMO convergence, decision-density, and scalar fitness values."""
    n = pop.objs.shape[0]
    _, r = _dominance_strength_fitness(pop.objs)
    k = int(math.floor(math.sqrt(n)))
    d_pop = _kth_density(pop.objs, k)
    d_dec = _kth_density(pop.decs, k)
    fitness = r + d_pop + d_dec
    return d_dec, d_pop, fitness


def _cmmo_dec_epsilon_fitness(pop: Population, epsilon: float | torch.Tensor):
    """Calculate CMMO decision fitness under an objective epsilon threshold."""
    obj = pop.objs
    diff = obj[None, :, :] - obj[:, None, :]
    eps = torch.as_tensor(epsilon, dtype=obj.dtype, device=obj.device)
    k = torch.any(diff > eps, dim=2).to(torch.int8) - torch.any(diff < eps, dim=2).to(torch.int8)
    dom = k == 1
    s = torch.sum(dom, dim=1).to(obj.dtype)
    r = torch.sum(dom.T.to(obj.dtype) * s.reshape(1, -1), dim=1)
    d_dec = _kth_density(pop.decs, int(math.floor(math.sqrt(obj.shape[0]))))
    return r + d_dec, d_dec


def _spea2_environmental(pop: Population, n: int, by_dec_epsilon: float | None = None):
    """Apply strength fitness and distance truncation to select survivors."""
    if by_dec_epsilon is None:
        d_dec, d_pop, fitness = _cmmo_fitness(pop)
    else:
        fitness, d_dec = _cmmo_dec_epsilon_fitness(pop, by_dec_epsilon)
        d_pop = None
    nxt = fitness < 1
    count = int(torch.sum(nxt).item())
    if count < n:
        rank = torch.argsort(fitness, stable=True)
        nxt[rank[:n]] = True
    elif count > n:
        sel = torch.where(nxt)[0]
        sel_pop = _subset(pop, sel)
        spaces = (sel_pop.decs,) if by_dec_epsilon is not None else (sel_pop.objs, sel_pop.decs)
        delete = _truncation_by_distance(*spaces, k=count - n)
        nxt[sel[delete]] = False
    chosen = torch.where(nxt)[0]
    chosen = chosen[torch.argsort(fitness[chosen], stable=True)]
    new_pop = _subset(pop, chosen)
    new_fit = fitness[chosen]
    new_dec = d_dec[chosen]
    new_pop_d = None if d_pop is None else d_pop[chosen]
    return new_pop, new_fit, new_dec, new_pop_d


def _hrea_environmental(pop: Population, n: int):
    """Select the HREA main population by hierarchy and neighborhood diversity."""
    local_c = _local_convergence(pop)
    short_dist = _sum_nearest_distance(pop.decs, 3)
    order = _lex_sort_indices(local_c, -short_dist)
    order = order[: min(n, order.numel())]
    new_pop = _subset(pop, order)
    return new_pop, _harmonic_crowding(new_pop.decs)


def _hrea_archive_update(pop: Population, n: int, eps: float, st: float):
    """Update the HREA archive with objective and decision-space niches."""
    if eps != 1 and st < 0.5:
        eps = 2 * (1 - eps) / (2 * st + 1) + 2 * eps - 1
    first_mask = nd_sort(pop.objs)[0] == 1
    first_pf = _subset(pop, torch.where(first_mask)[0])
    new_pop = first_pf
    remain = _subset(pop, torch.where(~first_mask)[0])
    if pop.decs.shape[0] > 0:
        span = torch.max(pop.decs, dim=0).values - torch.min(pop.decs, dim=0).values
        v = 0.2 * torch.prod(span).pow(1.0 / max(pop.decs.shape[1], 1))
    else:
        v = torch.tensor(0.0, dtype=pop.decs.dtype, device=pop.decs.device)
    while remain.decs.shape[0] > 0:
        dist = pairwise_dist(new_pop.decs, remain.decs).min(dim=0).values
        keep = dist >= v
        remain = _subset(remain, torch.where(keep)[0])
        if remain.decs.shape[0] == 0:
            break
        front_mask = nd_sort(remain.objs)[0] == 1
        pick_pop = _subset(remain, torch.where(front_mask)[0])
        combined_obj = torch.cat([pick_pop.objs * (1 - eps), first_pf.objs], dim=0)
        front_no = nd_sort(combined_obj)[0][: pick_pop.decs.shape[0]]
        if int(torch.max(front_no).item()) > 1:
            new_pop = _cat(new_pop, _subset(pick_pop, torch.where(front_no == 1)[0]))
            break
        new_pop = _cat(new_pop, pick_pop)
        remain = _subset(remain, torch.where(~front_mask)[0])
    if new_pop.decs.shape[0] > n:
        front_no = nd_sort(new_pop.objs)[0]
        max_front = int(torch.max(front_no).item())
        n_sub = int(math.ceil(n / max(max_front, 1)))
        selected_parts = []
        selected_weights = []
        tmp_parts = []
        for front in range(1, max_front + 1):
            idx = torch.where(front_no == front)[0]
            if idx.numel() < n_sub:
                selected_parts.append(_subset(new_pop, idx))
                selected_weights.append(
                    torch.full(
                        (idx.numel(),),
                        n_sub - idx.numel(),
                        dtype=new_pop.decs.dtype,
                        device=new_pop.decs.device,
                    )
                )
            else:
                tmp_parts.append(_subset(new_pop, idx))
        sel_pop = _cat(*selected_parts) if selected_parts else _empty_like(new_pop)
        tmp_pop = _cat(*tmp_parts) if tmp_parts else _empty_like(new_pop)
        while tmp_pop.decs.shape[0] > n - sel_pop.decs.shape[0]:
            crowd = _sum_nearest_distance(tmp_pop.decs, 3)
            keep = torch.ones(tmp_pop.decs.shape[0], dtype=torch.bool, device=tmp_pop.decs.device)
            keep[torch.argmin(crowd)] = False
            tmp_pop = _subset(tmp_pop, torch.where(keep)[0])
        new_pop = _cat(sel_pop, tmp_pop) if sel_pop.decs.shape[0] else tmp_pop
        archive_weight = (
            torch.cat(
                selected_weights
                + [
                    torch.zeros(
                        tmp_pop.decs.shape[0],
                        dtype=new_pop.decs.dtype,
                        device=new_pop.decs.device,
                    )
                ]
            )
            + 1
        )
        return new_pop, _harmonic_crowding(new_pop.decs) * archive_weight
    return new_pop, _harmonic_crowding(new_pop.decs)


def _commea_fitness(pop: Population, operation: str):
    """Calculate CoMMEA convergence and local-niche fitness."""
    local_mask = None
    if operation == "LocalC":
        dist = pairwise_dist(pop.decs, pop.decs)
        radius = torch.mean(dist) / (4.0 if pop.decs.shape[1] <= 8 else 2.0)
        local_mask = dist < radius
    _, fitness = _dominance_strength_fitness(pop.objs, local_mask)
    return fitness


def _commea_environmental1(pop: Population, n: int, operation: str, evo_state: float):
    """Select CoMMEA's global convergence population."""
    fitness = _commea_fitness(pop, operation)
    nxt = fitness < 1
    if int(torch.sum(nxt).item()) < n:
        order = _lex_sort_indices(fitness, -_harmonic_crowding(pop.decs))[:n]
        new_pop = _subset(pop, order)
    else:
        new_pop = _subset(pop, torch.where(nxt)[0])
        while new_pop.decs.shape[0] > n:
            crowd = _sum_nearest_distance(new_pop.decs, 3)
            keep = torch.ones(new_pop.decs.shape[0], dtype=torch.bool, device=new_pop.decs.device)
            keep[torch.argmin(crowd)] = False
            new_pop = _subset(new_pop, torch.where(keep)[0])
    if evo_state < 0.5:
        parent_fit = _commea_fitness(new_pop, operation) - _harmonic_crowding(new_pop.decs)
    else:
        parent_fit = -_harmonic_crowding(new_pop.decs)
    return new_pop, parent_fit


def _kdis(pop: Population, k: int | float):
    """Calculate kth-nearest decision-space distances."""
    n = pop.decs.shape[0]
    if n <= 1:
        return torch.ones(n, dtype=pop.decs.dtype, device=pop.decs.device)
    k = max(1, min(int(k), n - 1))
    dn_dec = _sum_nearest_distance(pop.decs, k + 1)
    avg = torch.mean(dn_dec)
    avg = torch.where(avg == 0, torch.tensor(float("inf"), dtype=avg.dtype, device=avg.device), avg)
    return 1.0 / (1.0 + dn_dec / avg)


def _commea_environmental2(pop: Population, n: int, operation: str, eps: float):
    """Select CoMMEA's local decision-diversity population."""
    front_no = nd_sort(pop.objs)[0]
    next_idx = torch.where(front_no == 1)[0]
    remain_idx = torch.where(front_no > 1)[0]
    if remain_idx.numel() > 0 and next_idx.numel() > 0:
        remain_obj = pop.objs[remain_idx]
        next_obj = pop.objs[next_idx]
        eps_obj = (1 + eps) * next_obj
        dominated = torch.all(eps_obj[None, :, :] <= remain_obj[:, None, :], dim=2) & torch.any(
            eps_obj[None, :, :] < remain_obj[:, None, :], dim=2
        )
        eps_sol = ~torch.any(dominated, dim=1)
    else:
        eps_sol = torch.zeros(remain_idx.numel(), dtype=torch.bool, device=pop.decs.device)
    if next_idx.numel() + int(torch.sum(eps_sol).item()) < n:
        crowd = -_harmonic_crowding(pop.decs)
        order = _lex_sort_indices(front_no, crowd)[:n]
        pop = _subset(pop, order)
    else:
        chosen = torch.cat([next_idx, remain_idx[eps_sol]])
        pop = _subset(pop, chosen)
    fitness = _commea_fitness(pop, operation)
    nxt = fitness < 1
    k_near = 3
    if int(torch.sum(nxt).item()) < n:
        dist = torch.sort(pairwise_dist(pop.decs, pop.decs), dim=0).values
        crowd = -torch.sum(dist[: min(k_near, dist.shape[0]), :], dim=0)
        order = _lex_sort_indices(fitness, crowd)[:n]
        pop = _subset(pop, order)
    else:
        pop = _subset(pop, torch.where(nxt)[0])
        while pop.decs.shape[0] > n:
            dobj = _sum_nearest_distance(pop.objs, k_near)
            ddec = _sum_nearest_distance(pop.decs, k_near)
            tiny = torch.finfo(pop.decs.dtype).tiny
            crowd = dobj / torch.max(dobj).clamp_min(tiny) + ddec / torch.max(
                ddec
            ).clamp_min(tiny)
            keep = torch.ones(pop.decs.shape[0], dtype=torch.bool, device=pop.decs.device)
            keep[torch.argmin(crowd)] = False
            pop = _subset(pop, torch.where(keep)[0])
    return pop, _kdis(pop, n / 2)
