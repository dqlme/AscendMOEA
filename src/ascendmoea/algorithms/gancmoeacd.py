"""Implementation helpers for the gancmoeacd algorithm."""

from __future__ import annotations

import math
import random

import torch
import torch.nn.functional as F
from torch import nn

from ..core import Algorithm, Population, Problem
from ..utils import merge_pop, operator_de, operator_ga, pairwise_dist, uniform_point
from .cmoeacd import (
    _diversity_enhancement_archive,
    _ensure_pop,
    _feasibility_exploitation_archive,
    _forward_exploration_archive,
    _subset,
    _truncation,
)


class _CondGenerator(nn.Module):
    """Implement the _CondGenerator multi-objective evolutionary algorithm."""

    def __init__(self, latent_dim: int, m: int, d: int, hidden_dim: int):
        """Initialize the instance."""
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim + m, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, d),
            nn.Sigmoid(),
        )

    def forward(self, z: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        """Compute forward."""
        return self.net(torch.cat([z, w], dim=1))


class _CondDiscriminator(nn.Module):
    """Implement the _CondDiscriminator multi-objective evolutionary algorithm."""

    def __init__(self, m: int, d: int, hidden_dim: int):
        """Initialize the instance."""
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(d + m, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
        )
        self.adv = nn.Linear(hidden_dim, 1)
        self.cv = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor, w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute forward."""
        h = self.shared(torch.cat([x, w], dim=1))
        return self.adv(h), F.softplus(self.cv(h)).squeeze(1)


def _cv(cons: torch.Tensor) -> torch.Tensor:
    """Compute cv."""
    return torch.sum(torch.clamp(cons, min=0.0), dim=1)


def _normalize_dec(problem: Problem, dec: torch.Tensor) -> torch.Tensor:
    """Compute normalize dec."""
    den = torch.clamp(problem.upper - problem.lower, min=1e-12)
    return torch.clamp((dec - problem.lower) / den, 0.0, 1.0)


def _denormalize_dec(problem: Problem, dec_norm: torch.Tensor) -> torch.Tensor:
    """Compute denormalize dec."""
    dec = problem.lower + dec_norm * (problem.upper - problem.lower)
    return torch.minimum(torch.maximum(dec, problem.lower), problem.upper)


def _reference_labels(objs: torch.Tensor, refs: torch.Tensor) -> torch.Tensor:
    """Compute reference labels."""
    zmin = torch.min(objs, dim=0, keepdim=True).values
    zmax = torch.max(objs, dim=0, keepdim=True).values
    obj_n = (objs - zmin) / torch.clamp(zmax - zmin, min=1e-12)
    obj_n = F.normalize(torch.clamp(obj_n, min=1e-12), p=2, dim=1)
    ref_n = F.normalize(refs, p=2, dim=1)
    return torch.argmax(torch.mm(obj_n, ref_n.t()), dim=1)


def _training_pool(
    problem: Problem,
    fa: Population,
    da: Population,
    fea: Population,
    refs: torch.Tensor,
    max_size: int,
) -> Population:
    """Compute training pool."""
    pools = [p for p in (fa, da, fea) if p.decs.shape[0] > 0]
    if not pools:
        return _ensure_pop(problem, None)
    pool = merge_pop(*pools)
    cv = _cv(pool.cons)
    labels = _reference_labels(pool.objs, refs)
    ref_n = F.normalize(refs, p=2, dim=1)
    obj_n = pool.objs - torch.min(pool.objs, dim=0, keepdim=True).values
    obj_n = obj_n / torch.clamp(torch.max(obj_n, dim=0, keepdim=True).values, min=1e-12)
    obj_n = F.normalize(torch.clamp(obj_n, min=1e-12), p=2, dim=1)
    sim = torch.sum(obj_n * ref_n[labels], dim=1)
    infeasible = (cv > 0).to(pool.objs.dtype)

    keep_parts = []
    for r in torch.unique(labels):
        idx = torch.where(labels == r)[0]
        local = idx[torch.argsort(-sim[idx], stable=True)]
        local = local[torch.argsort(cv[local], stable=True)]
        local = local[torch.argsort(infeasible[local], stable=True)]
        keep_parts.append(local[:1])

    keep = torch.cat(keep_parts, dim=0) if keep_parts else torch.argsort(cv)[:max_size]
    if keep.numel() < min(max_size, 8) and pool.decs.shape[0] > keep.numel():
        extra = torch.argsort(cv)[: min(max_size, pool.decs.shape[0])]
        keep = torch.unique(torch.cat([keep, extra], dim=0))
    if keep.numel() > max_size:
        local = keep[torch.argsort(-sim[keep], stable=True)]
        local = local[torch.argsort(cv[local], stable=True)]
        keep = local[torch.argsort(infeasible[local], stable=True)[:max_size]]
    return _subset(pool, keep)


def _pick_sparse_reference_vectors(
    pool: Population, refs: torch.Tensor, max_k: int
) -> torch.Tensor:
    """Compute pick sparse reference vectors."""
    if max_k <= 0:
        return refs[:0]
    if pool.decs.shape[0] == 0:
        return refs[torch.randperm(refs.shape[0], device=refs.device)[:max_k]]

    labels = _reference_labels(pool.objs, refs)
    counts = torch.bincount(labels, minlength=refs.shape[0]).to(pool.objs.dtype)

    empty = torch.where(counts == 0)[0]
    if empty.numel() > 0:
        return refs[empty[: min(max_k, empty.numel())]]
    return refs[:0]


def _pick_weak_reference_vectors(pool: Population, refs: torch.Tensor, max_k: int) -> torch.Tensor:
    """Compute pick weak reference vectors."""
    if max_k <= 0:
        return refs[:0]
    if pool.decs.shape[0] == 0:
        return refs[torch.randperm(refs.shape[0], device=refs.device)[:max_k]]

    labels = _reference_labels(pool.objs, refs)
    counts = torch.bincount(labels, minlength=refs.shape[0]).to(pool.objs.dtype)
    cv = _cv(pool.cons)
    feasible = (cv <= 0).to(pool.objs.dtype)
    feasible_counts = torch.zeros(refs.shape[0], dtype=pool.objs.dtype, device=refs.device)
    feasible_counts.scatter_add_(0, labels, feasible)
    cv_sum = torch.zeros(refs.shape[0], dtype=pool.objs.dtype, device=refs.device)
    cv_sum.scatter_add_(0, labels, cv)
    mean_cv = cv_sum / torch.clamp(counts, min=1.0)

    order = torch.argsort(-mean_cv, stable=True)
    order = order[torch.argsort(counts[order], stable=True)]
    order = order[torch.argsort(feasible_counts[order], stable=True)]
    return refs[order[: min(max_k, order.numel())]]


def _final_feasible_output(pop: Population, n: int) -> Population:
    """Compute final feasible output."""
    cv = _cv(pop.cons)
    feasible_idx = torch.where(cv <= 0)[0]
    if feasible_idx.numel() == 0:
        keep = torch.argsort(cv)[: min(n, pop.decs.shape[0])]
        out = _subset(pop, keep)
    else:
        out = _subset(pop, feasible_idx)

    if out.decs.shape[0] > n:
        obj = out.objs
        zmin = torch.min(obj, dim=0).values
        zmax = torch.max(obj, dim=0).values
        obj_n = (obj - zmin) / torch.clamp(zmax - zmin, min=1e-12)
        delete = _truncation(obj_n, out.decs.shape[0] - n)
        out = _subset(out, torch.where(~delete)[0])

    if out.decs.shape[0] < n and out.decs.shape[0] > 0:
        rep = torch.arange(n, device=out.decs.device) % out.decs.shape[0]
        out = _subset(out, rep)

    return out


def _lg(x: torch.Tensor, sigma: torch.Tensor | float) -> torch.Tensor:
    """Compute lg."""
    sigma_t = torch.as_tensor(sigma, dtype=x.dtype, device=x.device).clamp_min(1e-12)
    return torch.sigmoid(x / sigma_t)


def _fuzzy_decomposition_update(
    dpop: Population,
    cand: Population,
    refs: torch.Tensor,
    neighbors: torch.Tensor,
    z: torch.Tensor,
    nr: int,
) -> Population:
    """Compute fuzzy decomposition update."""
    if cand.decs.shape[0] == 0:
        return dpop

    obj_t = dpop.objs
    cv_t = _cv(dpop.cons)
    cand_obj = cand.objs
    cand_cv = _cv(cand.cons)
    w = torch.clamp(refs, min=1e-6)
    n = dpop.decs.shape[0]
    t = neighbors.shape[1]
    nr = max(1, int(nr))

    all_cv = torch.cat([cv_t, cand_cv], dim=0)
    con_min = torch.min(all_cv)
    con_max = torch.max(all_cv)
    cv_den = torch.clamp(con_max - con_min, min=1e-12)

    for j in range(cand.decs.shape[0]):
        p = neighbors[j % n, torch.randperm(t, device=refs.device)]
        old_obj = obj_t[p]
        new_obj = cand_obj[j]
        w_p = w[p]

        g_old = torch.max(torch.abs(old_obj - z) / w_p, dim=1).values
        g_new = torch.max(torch.abs(new_obj - z).unsqueeze(0) / w_p, dim=1).values

        cv_old = cv_t[p]
        cv_new = cand_cv[j]
        if con_max > con_min:
            cv_old_n = torch.where(cv_old > 0, (cv_old - con_min) / cv_den, cv_old)
            cv_new_n = (cv_new - con_min) / cv_den if cv_new > 0 else cv_new
        else:
            cv_old_n = cv_old
            cv_new_n = cv_new

        sigma_obj = torch.std(torch.cat([g_old, g_new], dim=0)).clamp_min(1e-12)
        sigma_cv = torch.std(torch.cat([cv_old_n, cv_new_n.reshape(1)], dim=0)).clamp_min(1e-12)
        new_old = _lg(g_old - g_new, sigma_obj) * torch.clamp(
            _lg(cv_old_n - cv_new_n, sigma_cv), min=1e-4
        )
        old_new = _lg(g_new - g_old, sigma_obj) * torch.clamp(
            _lg(cv_new_n - cv_old_n, sigma_cv), min=1e-4
        )
        rep = p[new_old >= old_new]
        if rep.numel() > nr:
            rep = rep[:nr]
        if rep.numel() > 0:
            obj_t[rep] = new_obj
            cv_t[rep] = cand_cv[j]
            dpop.decs[rep] = cand.decs[j]
            dpop.objs[rep] = cand.objs[j]
            dpop.cons[rep] = cand.cons[j]

    return dpop


class _ConditionalGAN:
    """Implement the _ConditionalGAN multi-objective evolutionary algorithm."""

    def __init__(
        self,
        problem: Problem,
        latent_dim: int,
        hidden_dim: int,
        lr: float,
        cv_weight: float,
    ):
        """Initialize the instance."""
        self.problem = problem
        self.latent_dim = int(latent_dim)
        self.cv_weight = float(cv_weight)
        self.gen = _CondGenerator(latent_dim, problem.M, problem.D, hidden_dim).to(
            device=problem.device, dtype=problem.dtype
        )
        self.dis = _CondDiscriminator(problem.M, problem.D, hidden_dim).to(
            device=problem.device, dtype=problem.dtype
        )
        self.opt_g = torch.optim.Adam(self.gen.parameters(), lr=lr, betas=(0.5, 0.999))
        self.opt_d = torch.optim.Adam(self.dis.parameters(), lr=lr, betas=(0.5, 0.999))

    def train_steps(
        self, pool: Population, refs: torch.Tensor, epochs: int, batch_size: int
    ) -> None:
        """Compute train steps."""
        n = pool.decs.shape[0]
        if n < 8:
            return

        x_all = _normalize_dec(self.problem, pool.decs).detach()
        cv_all = _cv(pool.cons).detach()
        cv_scale = torch.clamp(torch.max(cv_all), min=1.0)
        cv_all = cv_all / cv_scale
        labels = _reference_labels(pool.objs, refs)
        w_all = refs[labels].detach()
        bce = nn.BCEWithLogitsLoss()
        mse = nn.MSELoss()
        batch = min(max(8, int(batch_size)), n)

        for _ in range(max(1, int(epochs))):
            idx = torch.randint(0, n, (batch,), device=self.problem.device)
            real_x = x_all[idx]
            real_w = w_all[idx]
            real_cv = cv_all[idx]
            z = torch.randn(
                batch, self.latent_dim, dtype=self.problem.dtype, device=self.problem.device
            )
            fake_x = self.gen(z, real_w).detach()

            self.opt_d.zero_grad(set_to_none=True)
            real_logits, pred_cv = self.dis(real_x, real_w)
            fake_logits, _ = self.dis(fake_x, real_w)
            loss_d = (
                bce(real_logits, torch.ones_like(real_logits))
                + bce(fake_logits, torch.zeros_like(fake_logits))
                + self.cv_weight * mse(pred_cv, real_cv)
            )
            loss_d.backward()
            self.opt_d.step()

            z = torch.randn(
                batch, self.latent_dim, dtype=self.problem.dtype, device=self.problem.device
            )
            self.opt_g.zero_grad(set_to_none=True)
            gen_x = self.gen(z, real_w)
            gen_logits, gen_cv = self.dis(gen_x, real_w)
            diversity = torch.mean(torch.var(gen_x, dim=0))
            loss_g = (
                bce(gen_logits, torch.ones_like(gen_logits))
                + self.cv_weight * torch.mean(gen_cv)
                - 0.05 * diversity
            )
            loss_g.backward()
            self.opt_g.step()

    @torch.no_grad()
    def sample(self, refs: torch.Tensor) -> torch.Tensor:
        """Compute sample."""
        z = torch.randn(
            refs.shape[0], self.latent_dim, dtype=self.problem.dtype, device=self.problem.device
        )
        dec_norm = self.gen(z, refs)
        return _denormalize_dec(self.problem, dec_norm)

    @torch.no_grad()
    def sample_best_per_ref(self, refs: torch.Tensor, candidates_per_ref: int = 3) -> torch.Tensor:
        """Compute sample best per ref."""
        if refs.shape[0] == 0:
            return torch.empty(
                (0, self.problem.D), dtype=self.problem.dtype, device=self.problem.device
            )
        repeats = max(1, int(candidates_per_ref))
        ref_rep = refs.repeat_interleave(repeats, dim=0)
        z = torch.randn(
            ref_rep.shape[0], self.latent_dim, dtype=self.problem.dtype, device=self.problem.device
        )
        dec_norm = self.gen(z, ref_rep)
        _, pred_cv = self.dis(dec_norm, ref_rep)
        dec_norm = dec_norm.reshape(refs.shape[0], repeats, self.problem.D)
        pred_cv = pred_cv.reshape(refs.shape[0], repeats)
        idx = torch.argmin(pred_cv, dim=1)
        chosen = dec_norm[torch.arange(refs.shape[0], device=self.problem.device), idx]
        return _denormalize_dec(self.problem, chosen)


def _disc_filter_replace(
    problem: Problem,
    gan: _ConditionalGAN | None,
    dec_keep: torch.Tensor,
    dec_filter: torch.Tensor,
    refs: torch.Tensor,
) -> torch.Tensor:
    """Compute disc filter replace."""
    if gan is None or dec_filter.shape[0] == 0 or dec_keep.shape[0] == 0:
        return dec_filter
    with torch.no_grad():
        pool = torch.cat([dec_keep, dec_filter], dim=0)
        fake_obj_refs = refs[
            torch.randint(0, refs.shape[0], (pool.shape[0],), device=problem.device)
        ]
        x = _normalize_dec(problem, pool)
        _, pred_cv = gan.dis(x, fake_obj_refs)
        keep_score = pred_cv[: dec_keep.shape[0]]
        filt_score = pred_cv[dec_keep.shape[0] :]
        k = max(1, dec_filter.shape[0] // 5)
        bad = torch.argsort(filt_score, descending=True)[:k]
        good = torch.argsort(keep_score)[:k]
        dec_filter = dec_filter.clone()
        dec_filter[bad] = dec_keep[good]
    return dec_filter


class GANCMOEACD(Algorithm):
    """Implement the GANCMOEACD multi-objective evolutionary algorithm."""

    name = "GANCMOEACD"

    def __init__(
        self,
        e1: int = 1,
        e2: int = 1,
        gan_start: float = 0.2,
        gan_interval: int = 5,
        gan_epochs: int = 10,
        latent_dim: int = 32,
        hidden_dim: int = 128,
        gan_lr: float = 1e-3,
        cv_weight: float = 1.0,
        **kwargs,
    ):
        """Initialize the instance."""
        super().__init__(**kwargs)
        self.e1 = int(e1)
        self.e2 = int(e2)
        self.gan_start = float(gan_start)
        self.gan_interval = max(1, int(gan_interval))
        self.gan_epochs = max(1, int(gan_epochs))
        self.latent_dim = max(2, int(latent_dim))
        self.hidden_dim = max(16, int(hidden_dim))
        self.gan_lr = float(gan_lr)
        self.cv_weight = float(cv_weight)

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
        refs, reference_count = uniform_point(problem.N, problem.M, device=problem.device)
        dpop_index = torch.arange(reference_count, device=problem.device) % population.decs.shape[0]
        dpop = _subset(population, dpop_index)
        nr = max(1, math.ceil(reference_count / 100))
        dist_matrix = pairwise_dist(refs, refs)
        neighbour_count = min(reference_count, max(2, math.ceil(reference_count / 10)))
        neighbors = torch.argsort(dist_matrix, dim=1, stable=True)[:, :neighbour_count]
        z_dec = torch.min(population.objs, dim=0).values
        gan = None
        generation = 0

        while self.not_terminated(problem, fea, output, should_stop, should_pause):
            generation += 1
            zmin = torch.minimum(zmin, torch.min(off.objs, dim=0).values - 1e-6)

            fa = _forward_exploration_archive(fa, off, zmin, ns, self.e1)
            da = _diversity_enhancement_archive(da, off, zmin, ns)
            fea = _feasibility_exploitation_archive(fea, off, problem.N, self.e2)

            progress = problem.FE / max(1, problem.max_fe)
            pop1_base = fa if fa.decs.shape[0] > 0 else fea
            pop1 = pop1_base
            pop2_base = da if da.decs.shape[0] > 0 else fea
            pop2 = pop2_base
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
                dec1 = operator_de(problem, pop1.decs, pop1.decs[p11], pop1.decs[p12], 1, 0.5, 1, 1)
                dec2 = operator_de(problem, pop2.decs, pop2.decs[p21], pop2.decs[p22], 1, 0.5, 1, 1)
                dec3 = operator_de(problem, pop3.decs, pop3.decs[p31], pop3.decs[p32], 1, 0.5, 1, 1)
            else:
                dec1 = operator_ga(problem, pop1.decs[p11], 1, 20, 1, 1)
                dec2 = operator_ga(problem, pop2.decs[p21], 1, 20, 1, 1)
                dec3 = operator_ga(problem, pop3.decs[p31], 1, 20, 1, 1)

            if progress >= self.gan_start:
                if gan is None:
                    gan = _ConditionalGAN(
                        problem, self.latent_dim, self.hidden_dim, self.gan_lr, self.cv_weight
                    )
                pool = _training_pool(
                    problem, merge_pop(fa, dpop), da, fea, refs, max(4 * problem.N, 64)
                )
                if generation % self.gan_interval == 0:
                    gan.train_steps(pool, refs, self.gan_epochs, batch_size=min(problem.N, 64))

                coverage_pool = merge_pop(fea, dpop)
                k_gan = max(1, dec3.shape[0] // 5)
                sample_refs = _pick_weak_reference_vectors(coverage_pool, refs, k_gan)
                gan_n = sample_refs.shape[0]
                if gan_n > 0 and pool.decs.shape[0] >= 8:
                    gan_dec = gan.sample_best_per_ref(sample_refs)
                    dec3[:gan_n] = gan_dec
                dec3 = _disc_filter_replace(
                    problem, gan, torch.cat([dec1, dec2], dim=0), dec3, refs
                )

            o1 = problem.evaluation(dec1)
            o2 = problem.evaluation(dec2)
            o3 = problem.evaluation(dec3)
            offspring = [o1, o2, o3]

            off = merge_pop(*offspring)
            z_dec = torch.minimum(z_dec, torch.min(off.objs, dim=0).values)
            if progress >= self.gan_start:
                dpop = _fuzzy_decomposition_update(dpop, off, refs, neighbors, z_dec, nr)

        final = _final_feasible_output(merge_pop(fea, off), problem.N)
        if self.result:
            self.result[-1] = (problem.FE, final)
