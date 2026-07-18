"""Implementation helpers for the mtccmoeacd algorithm."""

from __future__ import annotations

import math
import random

import torch

from ..core import Algorithm, Population, Problem
from ..utils import merge_pop, operator_de, operator_ga
from .cmoeacd import (
    _crowding_distance_all,
    _diversity_enhancement_archive,
    _ensure_pop,
    _feasibility_exploitation_archive,
    _forward_exploration_archive,
    _subset,
    _truncation,
)


def _cv(cons: torch.Tensor) -> torch.Tensor:
    """Aggregate positive constraint violations for each individual."""
    return torch.sum(torch.clamp(cons, min=0.0), dim=1)


def _objective_task_selection(pop: Population, n: int) -> Population:
    """Select survivors for the objective-oriented decomposition task."""
    obj = pop.objs
    less_eq = obj[:, None, :] <= obj[None, :, :]
    less = obj[:, None, :] < obj[None, :, :]
    dom = torch.all(less_eq, dim=2) & torch.any(less, dim=2)
    nd = ~torch.any(dom, dim=0)
    pop = _subset(pop, torch.where(nd)[0])
    if pop.decs.shape[0] <= n:
        idx = torch.arange(n, device=pop.decs.device) % pop.decs.shape[0]
        return _subset(pop, idx)

    zmin = torch.min(pop.objs, dim=0).values
    zmax = torch.max(pop.objs, dim=0).values
    obj_n = (pop.objs - zmin) / torch.clamp(zmax - zmin, min=1e-12)
    delete = _truncation(obj_n, pop.decs.shape[0] - n)
    return _subset(pop, torch.where(~delete)[0])


def _boundary_task_selection(pop: Population, n: int) -> Population:
    """Select survivors for the feasibility-boundary task."""
    cv = _cv(pop.cons)
    order_cv = torch.argsort(cv)
    pre = _subset(pop, order_cv[: min(pop.decs.shape[0], max(n, 2 * n))])
    if pre.decs.shape[0] <= n:
        idx = torch.arange(n, device=pre.decs.device) % pre.decs.shape[0]
        return _subset(pre, idx)

    obj = pre.objs
    zmin = torch.min(obj, dim=0).values
    zmax = torch.max(obj, dim=0).values
    obj_n = (obj - zmin) / torch.clamp(zmax - zmin, min=1e-12)
    cd = _crowding_distance_all(obj_n)
    keep = torch.argsort(-cd)[:n]
    return _subset(pre, keep)


def _final_feasible_output(pop: Population, n: int) -> Population:
    """Return feasible nondominated solutions when available."""
    cv = _cv(pop.cons)
    feasible = torch.where(cv <= 0)[0]
    if feasible.numel() > 0:
        out = _subset(pop, feasible)
    else:
        out = _subset(pop, torch.argsort(cv)[: min(n, pop.decs.shape[0])])

    if out.decs.shape[0] > n:
        obj = out.objs
        zmin = torch.min(obj, dim=0).values
        zmax = torch.max(obj, dim=0).values
        obj_n = (obj - zmin) / torch.clamp(zmax - zmin, min=1e-12)
        delete = _truncation(obj_n, out.decs.shape[0] - n)
        out = _subset(out, torch.where(~delete)[0])

    if 0 < out.decs.shape[0] < n:
        idx = torch.arange(n, device=out.decs.device) % out.decs.shape[0]
        out = _subset(out, idx)
    return out


def _make_offspring(problem: Problem, pop: Population, n: int) -> Population:
    """Generate offspring for one task with optional migrated parents."""
    if pop.decs.shape[0] == 0:
        return _ensure_pop(problem, None)

    if pop.decs.shape[0] < n:
        idx = torch.arange(n, device=problem.device) % pop.decs.shape[0]
        base = _subset(pop, idx)
    else:
        idx = torch.randperm(pop.decs.shape[0], device=problem.device)[:n]
        base = _subset(pop, idx)

    p1 = torch.randperm(base.decs.shape[0], device=problem.device)
    p2 = torch.randperm(base.decs.shape[0], device=problem.device)
    if random.random() < 0.5:
        dec = operator_de(problem, base.decs, base.decs[p1], base.decs[p2], 1, 0.5, 1, 1)
    else:
        p = torch.randperm(base.decs.shape[0], device=problem.device)
        dec = operator_ga(problem, base.decs[p], 1, 20, 1, 1)
    return problem.evaluation(dec)


class MTCCMOEACD(Algorithm):
    """Exchange solutions between constrained decomposition search tasks."""

    name = "MTCCMOEACD"

    def __init__(
        self,
        e1: int = 1,
        e2: int = 1,
        migrate_rate: float = 0.05,
        feasible_gate: float = 0.2,
        **kwargs,
    ):
        """Configure archive selection and inter-task migration.

        Args:
            e1: Exploration archive mode: ``1`` distance, ``2`` crowding,
                or ``3`` reference-vector association.
            e2: Feasibility archive mode with the same values as ``e1``.
            migrate_rate: Fraction of candidates exchanged at migration events.
            feasible_gate: Feasible ratio required to enable directed migration.
            **kwargs: Arguments forwarded to :class:`~ascendmoea.Algorithm`.
        """
        super().__init__(**kwargs)
        self.e1 = int(e1)
        self.e2 = int(e2)
        self.migrate_rate = float(migrate_rate)
        self.feasible_gate = float(feasible_gate)
        if self.e1 not in {1, 2, 3} or self.e2 not in {1, 2, 3}:
            raise ValueError("e1 and e2 must each be 1, 2, or 3")
        if not 0 <= self.migrate_rate <= 1 or not 0 <= self.feasible_gate <= 1:
            raise ValueError("migrate_rate and feasible_gate must be between zero and one")

    def _solve(self, problem: Problem, output, should_stop, should_pause):
        """Execute the vectorized evolutionary loop."""
        torch.set_default_dtype(problem.dtype)

        ns = max(1, problem.N // 3)
        population = problem.initialization()
        fea = population
        obj_task = population
        boundary_task = population
        off = population
        zmin = torch.min(population.objs, dim=0).values - 1e-6
        fa = _ensure_pop(problem, None)
        da = _ensure_pop(problem, None)

        while self.not_terminated(
            problem, _final_feasible_output(fea, problem.N), output, should_stop, should_pause
        ):
            zmin = torch.minimum(zmin, torch.min(off.objs, dim=0).values - 1e-6)

            fa = _forward_exploration_archive(fa, off, zmin, ns, self.e1)
            da = _diversity_enhancement_archive(da, off, zmin, ns)
            fea = _feasibility_exploitation_archive(fea, off, problem.N, self.e2)
            obj_task = _objective_task_selection(merge_pop(obj_task, off), problem.N)
            boundary_task = _boundary_task_selection(merge_pop(boundary_task, off), problem.N)

            pop1 = fa if fa.decs.shape[0] > 0 else obj_task
            pop2 = da if da.decs.shape[0] > 0 else boundary_task
            n3 = max(1, problem.N // 3)
            mig = max(0, int(math.floor(n3 * self.migrate_rate + 0.5)))
            fea_rate = torch.mean((_cv(fea.cons) <= 0).to(problem.dtype)).item()
            if fea_rate < self.feasible_gate and mig > 0 and boundary_task.decs.shape[0] > 0:
                mig = min(mig, pop2.decs.shape[0], boundary_task.decs.shape[0])
                idx_base = torch.randperm(pop2.decs.shape[0], device=problem.device)[
                    : pop2.decs.shape[0] - mig
                ]
                idx_bnd = torch.randperm(boundary_task.decs.shape[0], device=problem.device)[:mig]
                pop2 = merge_pop(_subset(pop2, idx_base), _subset(boundary_task, idx_bnd))

            ridx = torch.randint(0, fea.decs.shape[0], (n3,), device=problem.device)
            pop3 = _subset(fea, ridx)

            o1 = _make_offspring(problem, pop1, pop1.decs.shape[0])
            o2 = _make_offspring(problem, pop2, pop2.decs.shape[0])
            o3 = _make_offspring(problem, pop3, pop3.decs.shape[0])
            off = merge_pop(o1, o2, o3)

        final = _final_feasible_output(merge_pop(fea, off), problem.N)
        if self.result:
            self.result[-1] = (problem.FE, final)
