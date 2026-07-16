"""Quality indicators for objective-space and decision-space approximation."""

from __future__ import annotations

import torch

from .core import Population, to_tensor
from .utils import pairwise_dist


def _best(pop: Population):
    """Compute best."""
    return pop.best()


def _cdist(a: torch.Tensor, b: torch.Tensor):
    """Compute cdist."""
    return pairwise_dist(a, b)


def igd(pop: Population, optimum) -> float:
    """Compute inverted generational distance (IGD).

    The indicator averages each reference point's distance to the nearest
    feasible nondominated solution. Lower values are better.
    """
    b = _best(pop).objs
    optimum = to_tensor(optimum, device=b.device, dtype=b.dtype)
    if b.shape[0] == 0 or b.shape[1] != optimum.shape[1]:
        return float("nan")
    return float(torch.mean(torch.min(_cdist(optimum, b), dim=1).values).item())


def gd(pop: Population, optimum) -> float:
    """Compute generational distance from the approximation to the reference."""
    b = _best(pop).objs
    optimum = to_tensor(optimum, device=b.device, dtype=b.dtype)
    if b.shape[0] == 0 or b.shape[1] != optimum.shape[1]:
        return float("nan")
    dist = torch.min(_cdist(b, optimum), dim=1).values
    return float((torch.linalg.norm(dist) / max(1, dist.numel())).item())


def igdp(pop: Population, optimum) -> float:
    """Compute the dominance-aware IGD+ indicator under minimization."""
    b = _best(pop).objs
    optimum = to_tensor(optimum, device=b.device, dtype=b.dtype)
    if b.shape[0] == 0 or b.shape[1] != optimum.shape[1]:
        return float("nan")
    diff = torch.clamp(b.unsqueeze(0) - optimum.unsqueeze(1), min=0.0)
    distance = torch.linalg.vector_norm(diff, dim=2)
    return float(torch.mean(torch.min(distance, dim=1).values).item())


def delta_p(pop: Population, optimum) -> float:
    """Compute the maximum of forward and inverse mean nearest distances."""
    b = _best(pop).objs
    optimum = to_tensor(optimum, device=b.device, dtype=b.dtype)
    if b.shape[0] == 0 or b.shape[1] != optimum.shape[1]:
        return float("nan")
    dist = _cdist(optimum, b)
    return float(
        max(
            torch.mean(torch.min(dist, dim=0).values).item(),
            torch.mean(torch.min(dist, dim=1).values).item(),
        )
    )


def igdx(pop: Population, pos) -> float:
    """Compute inverted generational distance in decision space (IGDX)."""
    dec = pop.decs
    pos = to_tensor(pos, device=dec.device, dtype=dec.dtype)
    if dec.shape[0] == 0 or dec.shape[1] != pos.shape[1]:
        return float("nan")
    return float(torch.mean(torch.min(_cdist(pos, dec), dim=1).values).item())


def coverage_rate(pop: Population, pos) -> float:
    """Compute geometric-mean decision-range coverage against a reference set."""
    dec = pop.decs
    pos = to_tensor(pos, device=dec.device, dtype=dec.dtype)
    if dec.shape[0] == 0 or dec.shape[1] != pos.shape[1]:
        return float("nan")

    reference_min = torch.min(pos, dim=0).values
    reference_max = torch.max(pos, dim=0).values
    obtained_min = torch.min(dec, dim=0).values
    obtained_max = torch.max(dec, dim=0).values
    constant = reference_max == reference_min
    disjoint = (obtained_min >= reference_max) | (reference_min >= obtained_max)
    denominator = torch.where(
        constant, torch.ones_like(reference_max), reference_max - reference_min
    )
    overlap = (
        torch.minimum(obtained_max, reference_max) - torch.maximum(obtained_min, reference_min)
    ) / denominator
    ratio = torch.where(constant, torch.ones_like(overlap), overlap)
    ratio = torch.where(disjoint & ~constant, torch.zeros_like(ratio), ratio)
    if torch.any(ratio <= 0):
        return 0.0
    return float(torch.exp(torch.mean(torch.log(ratio))).item())


def psp(pop: Population, pos) -> float:
    """Compute Pareto-set proximity as IGDX divided by coverage rate."""
    distance = igdx(pop, pos)
    coverage = coverage_rate(pop, pos)
    if coverage == 0:
        return float("inf")
    return float(distance / coverage)


def feasible_rate(pop: Population, _optimum=None) -> float:
    """Return the fraction of individuals satisfying every inequality constraint."""
    return float(torch.mean((torch.all(pop.cons <= 0, dim=1)).to(pop.cons.dtype)).item())


def dm(pop: Population, optimum) -> float:
    """Compute the diversity metric (DM) over occupied reference-grid cells."""
    p = _best(pop).objs
    optimum = to_tensor(optimum, device=p.device, dtype=p.dtype)
    if p.shape[0] == 0 or p.shape[1] != optimum.shape[1]:
        return float("nan")
    fmax = torch.max(optimum, dim=0).values
    fmin = torch.min(optimum, dim=0).values

    def cal_grid(arr, maxv, minv, div):
        """Compute cal grid."""
        n, m = arr.shape
        d = (maxv - minv) / max(div, 1)
        gloc = torch.ceil(
            (arr - minv.unsqueeze(0).expand(n, -1)) / d.unsqueeze(0).expand(n, -1)
        ).to(torch.long)
        gloc = torch.clamp(gloc, min=1)
        h = torch.zeros((m, div), dtype=torch.bool, device=arr.device)
        for i in range(m):
            h[i] = torch.isin(torch.arange(1, div + 1, device=arr.device), gloc[:, i])
        return h

    def cal_m(h, h_ref):
        """Compute cal m."""
        m = h.shape[0]
        ones = torch.ones((m, 1), dtype=torch.bool, device=h.device)
        h = torch.cat([ones, h, ones], dim=1)
        h_ref = torch.cat([ones, h_ref, ones], dim=1)
        val = 0.0
        for i in range(m):
            for j in range(1, h.shape[1] - 1):
                if not bool(h_ref[i, j]):
                    continue
                a, b, c = bool(h[i, j - 1]), bool(h[i, j]), bool(h[i, j + 1])
                if b and a and c:
                    val += 1.0
                elif b and (a or c):
                    val += 0.67
                elif b:
                    val += 0.75
                elif a and c:
                    val += 0.75
                elif a or c:
                    val += 0.5
        return val

    h_ref = cal_grid(optimum[:, :-1], fmax[:-1], fmin[:-1], p.shape[0])
    h = h_ref & cal_grid(p[:, :-1], fmax[:-1], fmin[:-1], p.shape[0])
    den = cal_m(h_ref, h_ref)
    return float(cal_m(h, h_ref) / den) if den > 0 else 0.0


def cpf(pop: Population, optimum) -> float:
    """Compute coverage over the mapped Pareto-front manifold (CPF)."""
    p = _best(pop).objs
    optimum = to_tensor(optimum, device=p.device, dtype=p.dtype)
    if p.shape[0] == 0 or p.shape[1] != optimum.shape[1]:
        return float("nan")

    def map_to_cube(x, pf):
        """Compute map to cube."""
        n, m = x.shape
        x = x - ((torch.sum(x, dim=1) - 1)[:, None] / m).expand(-1, m)
        pf = pf - ((torch.sum(pf, dim=1) - 1)[:, None] / m).expand(-1, m)
        x = x - torch.min(pf, dim=0).values.unsqueeze(0).expand(x.shape[0], -1)
        x = x / (torch.sum(x, dim=1)[:, None].expand(-1, m))
        x = torch.clamp(x, min=1e-6)
        c = torch.ones((n, m), dtype=x.dtype, device=x.device)
        for j in range(1, m):
            temp = x[:, j] / x[:, 0] * torch.prod(c[:, m - j : m - 1], dim=1)
            c[:, m - j - 1] = 1 / (temp + 1)
        y = c[:, : m - 1]
        pw = torch.arange(m - 1, 0, -1, dtype=x.dtype, device=x.device)
        return y**pw

    def coverage(arr, maxv):
        """Compute coverage."""
        n, m = arr.shape
        distance = torch.max(torch.abs(arr.unsqueeze(1) - arr.unsqueeze(0)), dim=2).values
        distance.fill_diagonal_(float("inf"))
        side_length = torch.min(distance, dim=1).values
        side_length = torch.minimum(
            side_length,
            torch.tensor(maxv, dtype=arr.dtype, device=arr.device) ** (1 / m),
        )
        lower = torch.maximum(
            torch.tensor(0.0, dtype=arr.dtype, device=arr.device),
            arr - (side_length / 2)[:, None].expand(-1, m),
        )
        upper = torch.minimum(
            torch.tensor(1.0, dtype=arr.dtype, device=arr.device),
            arr + (side_length / 2)[:, None].expand(-1, m),
        )
        return torch.sum(torch.prod(upper - lower, dim=1))

    if optimum.shape[0] > 1:
        fmin = torch.min(optimum, dim=0).values
        fmax = torch.max(optimum, dim=0).values
        p = (p - fmin) / (fmax - fmin)
        optimum = (optimum - fmin) / (fmax - fmin)
        close = torch.argmin(_cdist(p, optimum), dim=1)
        p = optimum[close]
        vpf = coverage(map_to_cube(optimum, optimum), float("inf"))
        v = coverage(map_to_cube(p, optimum), (vpf / p.shape[0]).item())
        return float((v / vpf).item())

    fmin = torch.min(p, dim=0).values
    fmax = torch.max(p, dim=0).values
    p = (p - fmin) / (fmax - fmin)
    return float(coverage(map_to_cube(p, p), 1 / p.shape[0]).item())


def hv(pop: Population, optimum) -> float:
    """Compute normalized dominated hypervolume (HV).

    Exact slicing is used for two and three objectives. Four or more objectives
    use one million Monte Carlo samples generated on the population device.
    Higher values are better.
    """
    p = _best(pop).objs
    optimum = to_tensor(optimum, device=p.device, dtype=p.dtype)
    if p.shape[0] == 0 or p.shape[1] != optimum.shape[1]:
        return float("nan")
    fmin = torch.minimum(
        torch.min(p, dim=0).values, torch.zeros(p.shape[1], dtype=p.dtype, device=p.device)
    )
    fmax = torch.max(optimum, dim=0).values
    p = (p - fmin) / ((fmax - fmin) * 1.1)
    p = p[torch.all(p <= 1, dim=1)]
    if p.shape[0] == 0:
        return 0.0
    m = p.shape[1]
    ref = torch.ones(m, dtype=p.dtype, device=p.device)

    if m >= 4:
        sample_num = 1000000
        maxv = ref
        minv = torch.min(p, dim=0).values
        samples = torch.rand((sample_num, m), dtype=p.dtype, device=p.device) * (maxv - minv) + minv
        alive = torch.ones(sample_num, dtype=torch.bool, device=p.device)
        for i in range(p.shape[0]):
            dom = torch.all(p[i] <= samples[alive], dim=1)
            idx = torch.where(alive)[0]
            alive[idx[dom]] = False
        return float(torch.prod(maxv - minv).item() * (1 - torch.mean(alive.to(p.dtype)).item()))

    def area_2d(points: torch.Tensor) -> torch.Tensor:
        """Compute area 2d."""
        order = torch.argsort(points[:, 0], stable=True)
        values = points[order]
        next_x = torch.cat([values[1:, 0], ref[:1]])
        min_y = torch.cummin(values[:, 1], dim=0).values
        return torch.sum(
            torch.clamp(next_x - values[:, 0], min=0.0) * torch.clamp(1 - min_y, min=0.0)
        )

    if m == 2:
        return float(area_2d(p).item())

    order = torch.argsort(p[:, 0], stable=True)
    points = p[order]
    next_x = torch.cat([points[1:, 0], ref[:1]])
    volume = torch.zeros((), dtype=p.dtype, device=p.device)
    for index in range(points.shape[0]):
        width = torch.clamp(next_x[index] - points[index, 0], min=0.0)
        volume = volume + width * area_2d(points[: index + 1, 1:])
    return float(volume.item())


METRICS = {
    "IGD": (igd, "min"),
    "HV": (hv, "max"),
    "GD": (gd, "min"),
    "IGDp": (igdp, "min"),
    "DeltaP": (delta_p, "min"),
    "DM": (dm, "max"),
    "CPF": (cpf, "max"),
    "IGDX": (igdx, "min"),
    "CR": (coverage_rate, "max"),
    "PSP": (psp, "min"),
    "Feasible_rate": (feasible_rate, "max"),
    "runtime": (None, "min"),
}

__all__ = [
    "METRICS",
    "coverage_rate",
    "cpf",
    "delta_p",
    "dm",
    "feasible_rate",
    "gd",
    "hv",
    "igd",
    "igdp",
    "igdx",
    "psp",
]
