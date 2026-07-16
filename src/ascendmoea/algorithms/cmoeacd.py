"""Implementation helpers for the cmoeacd algorithm."""

from __future__ import annotations

import random

import torch

from ..core import Algorithm, Population, Problem, round_half_away_from_zero
from ..utils import crowding_distance, merge_pop, operator_de, operator_ga, uniform_point
from .spea_common import truncation


def _subset(pop: Population, idx: torch.Tensor) -> Population:
    """Compute subset."""
    idx = idx.to(torch.long)
    return Population(pop.decs[idx], pop.objs[idx], pop.cons[idx])


def _ensure_pop(problem: Problem, pop: Population | None) -> Population:
    """Compute ensure pop."""
    if pop is not None:
        return pop
    return Population(
        torch.empty((0, problem.D), dtype=problem.dtype, device=problem.device),
        torch.empty((0, problem.M), dtype=problem.dtype, device=problem.device),
        torch.empty((0, 1), dtype=problem.dtype, device=problem.device),
    )


def _concat(a: Population, b: Population) -> Population:
    """Compute concat."""
    if a.decs.shape[0] == 0:
        return b
    if b.decs.shape[0] == 0:
        return a
    return merge_pop(a, b)


def _cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Return pairwise cosine similarities with zero-vector protection."""
    tiny = torch.finfo(a.dtype).tiny
    a = a / torch.linalg.vector_norm(a, dim=1, keepdim=True).clamp_min(tiny)
    b = b / torch.linalg.vector_norm(b, dim=1, keepdim=True).clamp_min(tiny)
    return torch.clamp(a @ b.T, min=-1, max=1)


def _truncation(pop_obj: torch.Tensor, k: int) -> torch.Tensor:
    """Apply stable lexicographic distance truncation."""
    return truncation(pop_obj, k)


def _crowding_distance_all(pop_obj: torch.Tensor) -> torch.Tensor:
    """Compute crowding distance all."""
    front = torch.ones(pop_obj.shape[0], dtype=pop_obj.dtype, device=pop_obj.device)
    return crowding_distance(pop_obj, front)


def _domination_cal(pop: Population, add: int) -> torch.Tensor:
    """Compute domination cal."""
    pop_obj = round_half_away_from_zero(pop.objs * 1e10) / 1e10
    cons = torch.sum(torch.clamp(pop.cons, min=0.0), dim=1)

    less_eq = pop_obj[:, None, :] <= pop_obj[None, :, :]
    less = pop_obj[:, None, :] < pop_obj[None, :, :]
    obj_eq = torch.all(pop_obj[:, None, :] == pop_obj[None, :, :], dim=2)
    obj_dom = torch.all(less_eq, dim=2) & torch.any(less, dim=2)

    if add == 0:
        earlier = torch.triu(torch.ones_like(obj_eq), diagonal=1)
        dom = obj_dom | (obj_eq & earlier)
    else:
        cv_p = cons[:, None]
        cv_q = cons[None, :]
        dom = (obj_eq & (cv_p <= cv_q)) | (obj_dom & ((cv_p <= 0) | (cv_p <= cv_q)))

    mutual = dom & dom.T
    dom[mutual] = False

    # Resolve mutual dominance ties with a triangular mask and no index matrix.
    dom[torch.triu(mutual, diagonal=1)] = True
    return ~torch.any(dom, dim=0)


def _forward_exploration_archive(
    fa: Population, off: Population, zmin: torch.Tensor, ns: int, mode: int
) -> Population:
    """Compute forward exploration archive."""
    fa = _concat(fa, off)
    nd = _domination_cal(fa, 0)
    fa = _subset(fa, torch.where(nd)[0])
    pop_obj = fa.objs
    p, m = pop_obj.shape

    if p <= ns:
        add_idx = torch.randint(0, p, (max(0, ns - p),), device=pop_obj.device)
        return _subset(fa, torch.cat([torch.arange(p, device=pop_obj.device), add_idx], dim=0))

    zmax = torch.max(pop_obj, dim=0).values
    obj_n = (pop_obj - zmin) / (zmax - zmin).clamp_min(torch.finfo(pop_obj.dtype).eps)

    if mode == 1:
        delete = _truncation(obj_n, p - ns)
        return _subset(fa, torch.where(~delete)[0])
    if mode == 2:
        cd = _crowding_distance_all(obj_n)
        return _subset(fa, torch.argsort(-cd, stable=True)[:ns])

    w, ns = uniform_point(ns, m, device=pop_obj.device)
    distance = torch.sqrt(torch.sum(obj_n**2, dim=1))

    # Evaluate sin(arccos(cosine)) through its stable algebraic identity.
    sim = _cosine_similarity(w, obj_n)
    angle_sin = torch.sqrt(torch.clamp(1.0 - sim**2, min=0.0))
    fit_matrix = distance.unsqueeze(0) * angle_sin
    chosen = torch.argmin(fit_matrix, dim=1)
    return _subset(fa, chosen)


def _feasibility_exploitation_archive(
    fea: Population, off: Population, n: int, mode: int
) -> Population:
    """Compute feasibility exploitation archive."""
    fea = _concat(fea, off)
    nd = _domination_cal(fea, 1)
    fea = _subset(fea, torch.where(nd)[0])
    pop_obj = fea.objs
    p, m = pop_obj.shape

    if p <= n:
        add_idx = torch.randint(0, p, (max(0, n - p),), device=pop_obj.device)
        return _subset(fea, torch.cat([torch.arange(p, device=pop_obj.device), add_idx], dim=0))

    cons = torch.sum(torch.clamp(fea.cons, min=0.0), dim=1)
    if int(torch.sum(cons <= 0).item()) <= n:
        return _subset(fea, torch.argsort(cons, stable=True)[:n])

    fea = _subset(fea, torch.where(cons <= 0)[0])
    pop_obj = fea.objs
    zmin = torch.min(pop_obj, dim=0).values - 1e-6
    zmax = torch.max(pop_obj, dim=0).values
    obj_n = (pop_obj - zmin) / (zmax - zmin).clamp_min(torch.finfo(pop_obj.dtype).eps)

    if mode == 1:
        delete = _truncation(obj_n, fea.decs.shape[0] - n)
        return _subset(fea, torch.where(~delete)[0])
    if mode == 2:
        cd = _crowding_distance_all(obj_n)
        return _subset(fea, torch.argsort(-cd, stable=True)[:n])

    w, n = uniform_point(n, m, device=pop_obj.device)
    distance = torch.sqrt(torch.sum(obj_n**2, dim=1))

    # Evaluate sin(arccos(cosine)) through its stable algebraic identity.
    sim = _cosine_similarity(w, obj_n)
    angle_sin = torch.sqrt(torch.clamp(1.0 - sim**2, min=0.0))
    fit_matrix = distance.unsqueeze(0) * angle_sin
    chosen = torch.argmin(fit_matrix, dim=1)
    return _subset(fea, chosen)


def _diversity_enhancement_archive(
    da: Population, off: Population, zmin: torch.Tensor, ns: int
) -> Population:
    """Compute diversity enhancement archive."""
    s = _concat(da, off)
    nd = _domination_cal(s, 1)
    s = _subset(s, torch.where(nd)[0])
    obj = s.objs
    cv = torch.sum(torch.clamp(s.cons, min=0.0), dim=1)
    _, m = obj.shape
    w, ns = uniform_point(ns, m, device=obj.device)

    sim_ww = _cosine_similarity(w, w)
    angle_ww = torch.arccos(torch.clamp(sim_ww, -1.0, 1.0))
    angle_ww.fill_diagonal_(float("inf"))
    h = torch.mean(torch.min(angle_ww, dim=1).values)

    if torch.any(cv <= 0):
        zmax = torch.max(obj[cv <= 0], dim=0).values
    else:
        zmax = obj[torch.argmin(cv)]

    obj_n = (obj - zmin) / (zmax - zmin).clamp_min(torch.finfo(obj.dtype).eps)

    sim_sw = _cosine_similarity(w, obj_n)
    # Compare sin(arccos(cosine)) with the angular threshold.
    angle_sw = torch.sqrt(torch.clamp(1 - sim_sw**2, min=0))

    lst_mask = angle_sw <= h
    no_true_rows = ~torch.any(lst_mask, dim=1)
    min_angle_idx = torch.argmin(angle_sw, dim=1)
    lst_mask[no_true_rows, min_angle_idx[no_true_rows]] = True

    cv_matrix = torch.where(
        lst_mask,
        cv.unsqueeze(0).expand_as(lst_mask),
        torch.full_like(lst_mask, float("inf"), dtype=obj.dtype),
    )
    feas_mask = cv_matrix <= 0
    has_feas = torch.any(feas_mask, dim=1)
    idx_nofeas = torch.argmin(cv_matrix, dim=1)
    angle_matrix = torch.where(feas_mask, angle_sw, torch.full_like(angle_sw, float("inf")))
    idx_feas = torch.argmin(angle_matrix, dim=1)
    chosen = torch.where(has_feas, idx_feas, idx_nofeas)
    return _subset(s, chosen)


class CMOEACD(Algorithm):
    """Coordinate exploration, feasibility, and diversity archives."""

    name = "CMOEACD"

    def __init__(self, e1: int = 1, e2: int = 1, **kwargs):
        """Configure environmental selection for two external archives.

        Args:
            e1: Exploration archive mode: ``1`` distance, ``2`` crowding,
                or ``3`` reference-vector association.
            e2: Feasibility archive mode with the same values as ``e1``.
            **kwargs: Arguments forwarded to :class:`~ascendmoea.Algorithm`.
        """
        super().__init__(**kwargs)
        self.e1 = int(e1)
        self.e2 = int(e2)
        if self.e1 not in {1, 2, 3} or self.e2 not in {1, 2, 3}:
            raise ValueError("e1 and e2 must each be 1, 2, or 3")

    def _solve(self, problem: Problem, output, should_stop, should_pause):
        """Execute the vectorized evolutionary loop."""
        torch.set_default_dtype(problem.dtype)
        ns = max(1, problem.N // 3)
        population = problem.initialization()
        fea = population
        off = population
        zmin = torch.min(population.objs, dim=0).values - 1e-6
        fa = _ensure_pop(problem, None)
        da = _ensure_pop(problem, None)

        while self.not_terminated(problem, fea, output, should_stop, should_pause):
            zmin = torch.minimum(zmin, torch.min(off.objs, dim=0).values - 1e-6)

            fa = _forward_exploration_archive(fa, off, zmin, ns, self.e1)
            da = _diversity_enhancement_archive(da, off, zmin, ns)
            fea = _feasibility_exploitation_archive(fea, off, problem.N, self.e2)

            pop1 = fa if fa.decs.shape[0] > 0 else fea
            pop2 = da if da.decs.shape[0] > 0 else fea
            n3 = max(1, problem.N // 3)

            ridx = torch.randint(0, fea.decs.shape[0], (n3,), device=problem.device)
            pop3 = _subset(fea, ridx)

            p11 = torch.randperm(pop1.decs.shape[0], device=problem.device)
            p12 = torch.randperm(pop1.decs.shape[0], device=problem.device)
            p21 = torch.randperm(pop2.decs.shape[0], device=problem.device)
            p22 = torch.randperm(pop2.decs.shape[0], device=problem.device)
            p31 = torch.randperm(pop3.decs.shape[0], device=problem.device)
            p32 = torch.randperm(pop3.decs.shape[0], device=problem.device)

            if random.random() < 0.5:
                o1 = problem.evaluation(
                    operator_de(problem, pop1.decs, pop1.decs[p11], pop1.decs[p12], 1, 0.5, 1, 1)
                )
                o2 = problem.evaluation(
                    operator_de(problem, pop2.decs, pop2.decs[p21], pop2.decs[p22], 1, 0.5, 1, 1)
                )
                o3 = problem.evaluation(
                    operator_de(problem, pop3.decs, pop3.decs[p31], pop3.decs[p32], 1, 0.5, 1, 1)
                )
            else:
                o1 = problem.evaluation(operator_ga(problem, pop1.decs[p11], 1, 20, 1, 1))
                o2 = problem.evaluation(operator_ga(problem, pop2.decs[p21], 1, 20, 1, 1))
                o3 = problem.evaluation(operator_ga(problem, pop3.decs[p31], 1, 20, 1, 1))

            off = merge_pop(o1, o2, o3)
