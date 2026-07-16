"""Vectorized implementation of the dascmop problem."""

from __future__ import annotations

import math

import numpy as np
import torch

from ..core import Problem
from ..utils import uniform_point


def _constraint_2obj(
    x1: torch.Tensor, sum1: torch.Tensor, mode: int, diff: tuple[float, float, float]
) -> torch.Tensor:
    """Compute constraint 2obj."""
    a = 20.0
    b = 2 * diff[0] - 1
    d = 0.0 if diff[1] == 0.0 else 0.5
    e = (
        d - torch.log(torch.tensor(diff[1], dtype=x1.dtype, device=x1.device))
        if diff[1] != 0
        else torch.tensor(1e30, dtype=x1.dtype, device=x1.device)
    )
    r = 0.5 * diff[2]
    if mode in (1, 4):
        f2 = 1 - x1**2 + sum1
    elif mode in (2, 5):
        f2 = 1 - torch.sqrt(x1) + sum1
    else:
        f2 = 1 - torch.sqrt(x1) + 0.5 * torch.abs(torch.sin(5 * torch.pi * x1)) + sum1
    f1 = x1 + sum1
    c1 = b - torch.sin(a * torch.pi * x1)
    c2 = -(e - sum1) * (sum1 - d)
    if diff[1] == 1.0:
        c2 = 1e-4 - torch.abs(sum1 - e)
    p_k = torch.tensor(
        [0.0, 1.0, 0.0, 1.0, 2.0, 0.0, 1.0, 2.0, 3.0], dtype=x1.dtype, device=x1.device
    )
    q_k = torch.tensor(
        [1.5, 0.5, 2.5, 1.5, 0.5, 3.5, 2.5, 1.5, 0.5], dtype=x1.dtype, device=x1.device
    )
    theta = torch.tensor(-0.25 * torch.pi, dtype=x1.dtype, device=x1.device)
    cth = torch.cos(theta)
    sth = torch.sin(theta)
    xx = (f1[:, None] - p_k) * cth - (f2[:, None] - q_k) * sth
    yy = (f1[:, None] - p_k) * sth + (f2[:, None] - q_k) * cth
    c3 = r - (xx**2) / 0.3 - (yy**2) / 1.2
    return torch.column_stack([c1, c2, c3])


def _constraint_3obj(
    x1: torch.Tensor,
    x2: torch.Tensor,
    sum1: torch.Tensor,
    mode: int,
    diff: tuple[float, float, float],
) -> torch.Tensor:
    """Compute constraint 3obj."""
    a = 20.0
    b = 2 * diff[0] - 1
    d = 0.0 if diff[1] == 0.0 else 0.5
    e = (
        d - torch.log(torch.tensor(diff[1], dtype=x1.dtype, device=x1.device))
        if diff[1] != 0
        else torch.tensor(1e30, dtype=x1.dtype, device=x1.device)
    )
    r = 0.5 * diff[2]
    if mode == 7:
        f1 = x1 * x2 + sum1
        f2 = x2 * (1 - x1) + sum1
        f3 = 1 - x2 + sum1
    else:
        f1 = torch.cos(0.5 * torch.pi * x1) * torch.cos(0.5 * torch.pi * x2) + sum1
        f2 = torch.cos(0.5 * torch.pi * x1) * torch.sin(0.5 * torch.pi * x2) + sum1
        f3 = torch.sin(0.5 * torch.pi * x1) + sum1
    c1 = b - torch.sin(a * torch.pi * x1)
    c2 = b - torch.cos(a * torch.pi * x2)
    c3 = -(e - sum1) * (sum1 - d)
    if diff[1] == 1.0:
        c3 = 1e-4 - torch.abs(sum1 - e)
    center = 1.0 / math.sqrt(3.0)
    xk = torch.tensor([1.0, 0.0, 0.0, center], dtype=x1.dtype, device=x1.device)
    yk = torch.tensor([0.0, 1.0, 0.0, center], dtype=x1.dtype, device=x1.device)
    zk = torch.tensor([0.0, 0.0, 1.0, center], dtype=x1.dtype, device=x1.device)
    c4 = r * r - (f1[:, None] - xk) ** 2 - (f2[:, None] - yk) ** 2 - (f3[:, None] - zk) ** 2
    return torch.column_stack([c1, c2, c3, c4])


def _constraint_2obj_numpy(
    x1: np.ndarray, sum1: np.ndarray, mode: int, diff: tuple[float, float, float]
) -> np.ndarray:
    """Compute constraint 2obj numpy."""
    a = 20.0
    b = 2 * diff[0] - 1
    d = 0.0 if diff[1] == 0.0 else 0.5
    e = d - np.log(diff[1]) if diff[1] != 0 else 1e30
    radius = 0.5 * diff[2]
    if mode in (1, 4):
        f2 = 1 - x1**2 + sum1
    elif mode in (2, 5):
        f2 = 1 - np.sqrt(x1) + sum1
    else:
        f2 = 1 - np.sqrt(x1) + 0.5 * np.abs(np.sin(5 * np.pi * x1)) + sum1
    f1 = x1 + sum1
    c1 = b - np.sin(a * np.pi * x1)
    c2 = -(e - sum1) * (sum1 - d)
    if diff[1] == 1.0:
        c2 = 1e-4 - np.abs(sum1 - e)
    p_k = np.array([0.0, 1.0, 0.0, 1.0, 2.0, 0.0, 1.0, 2.0, 3.0])
    q_k = np.array([1.5, 0.5, 2.5, 1.5, 0.5, 3.5, 2.5, 1.5, 0.5])
    theta = -0.25 * np.pi
    xx = (f1[:, None] - p_k) * np.cos(theta) - (f2[:, None] - q_k) * np.sin(theta)
    yy = (f1[:, None] - p_k) * np.sin(theta) + (f2[:, None] - q_k) * np.cos(theta)
    c3 = radius - xx**2 / 0.3 - yy**2 / 1.2
    return np.column_stack((c1, c2, c3))


def _reference_dascmop_1_or_4(n: int, idx: int) -> np.ndarray:
    """Compute reference dascmop 1 or 4."""
    x = np.arange(n, dtype=np.float64) / max(n - 1, 1)
    reference = np.column_stack((x, 1 - x**2)) + 0.5
    x1 = (np.sqrt(1 - 4 * (-reference[:, 0] + reference[:, 1] - 1)) - 1) / 2
    sum1 = reference[:, 0] - x1
    diff = (0.0, 0.5, 0.5) if idx == 1 else (0.5, 0.5, 0.5)
    constraints = _constraint_2obj_numpy(x1, sum1, idx, diff)
    reference = reference[np.all(constraints <= 0, axis=1)]
    return np.vstack((reference, np.array([[1.5, 0.5]])))


def _estimate_x1_das36(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Compute estimate x1 das36."""
    target = (x - y).reshape(-1)
    a = torch.linspace(0.0, 1.0, 10001, dtype=x.dtype, device=x.device)
    curve = a - 1.0 + torch.sqrt(a) - 0.5 * torch.abs(torch.sin(5 * torch.pi * a))
    out = torch.empty_like(target)
    chunk = 4096
    for s in range(0, target.numel(), chunk):
        t = target[s : s + chunk]
        idx = torch.argmin(torch.abs(t[:, None] - curve[None, :]), dim=1)
        out[s : s + chunk] = a[idx]
    return out.reshape_as(x)


class DASCMOP(Problem):
    """Define the DASCMOP benchmark with batched tensor evaluation."""

    def __init__(self, idx: int, n=100, m=None, d=30, max_fe=10000):
        """Initialize the suite problem and apply its fixed configuration."""
        self.idx = int(idx)
        self.name = f"DASCMOP{self.idx}"
        super().__init__(n=n, m=2 if idx <= 6 else 3, d=d, max_fe=max_fe)

    def setting(self):
        """Configure problem dimensions, bounds, encoding, and static data."""
        self.M = 2 if self.idx <= 6 else 3
        self.lower = torch.zeros(self.D, dtype=torch.float64, device=self.device)
        self.upper = torch.ones(self.D, dtype=torch.float64, device=self.device)

    def evaluation(self, dec):
        """Repair and evaluate a complete decision batch."""
        x = self._prepare_dec(dec)
        if x.shape[0] == 0:
            return self._empty_population()
        idx = self.idx
        if idx <= 3:
            s = torch.sum((x - torch.sin(0.5 * torch.pi * x[:, [0]])) ** 2, dim=1)
            f1 = x[:, 0] + s
            if idx == 1:
                f2 = 1 - x[:, 0] ** 2 + s
            elif idx == 2:
                f2 = 1 - torch.sqrt(x[:, 0]) + s
            else:
                f2 = (
                    1 - torch.sqrt(x[:, 0]) + 0.5 * torch.abs(torch.sin(5 * torch.pi * x[:, 0])) + s
                )
            obj = torch.column_stack([f1, f2])
            diff = (0.0, 0.5, 0.5) if idx <= 2 else (0.5, 0.5, 0.5)
            con = _constraint_2obj(x[:, 0], s, idx, diff)
        elif idx <= 6:
            xx = x[:, 1:]
            g1 = torch.sum((xx - 0.5) * (xx - 0.5) - torch.cos(20.0 * torch.pi * (xx - 0.5)), dim=1)
            s = self.D - 1 + g1
            f1 = x[:, 0] + s
            if idx == 4:
                f2 = 1 - x[:, 0] ** 2 + s
            elif idx == 5:
                f2 = 1 - torch.sqrt(x[:, 0]) + s
            else:
                f2 = (
                    1 - torch.sqrt(x[:, 0]) + 0.5 * torch.abs(torch.sin(5 * torch.pi * x[:, 0])) + s
                )
            obj = torch.column_stack([f1, f2])
            con = _constraint_2obj(x[:, 0], s, idx, (0.5, 0.5, 0.5))
        elif idx in (7, 8):
            xx = x[:, 2:]
            g1 = torch.sum((xx - 0.5) * (xx - 0.5) - torch.cos(20.0 * torch.pi * (xx - 0.5)), dim=1)
            s = self.D - 2 + g1
            if idx == 7:
                f1 = x[:, 0] * x[:, 1] + s
                f2 = x[:, 1] * (1 - x[:, 0]) + s
                f3 = 1 - x[:, 1] + s
            else:
                f1 = torch.cos(0.5 * torch.pi * x[:, 0]) * torch.cos(0.5 * torch.pi * x[:, 1]) + s
                f2 = torch.cos(0.5 * torch.pi * x[:, 0]) * torch.sin(0.5 * torch.pi * x[:, 1]) + s
                f3 = torch.sin(0.5 * torch.pi * x[:, 0]) + s
            obj = torch.column_stack([f1, f2, f3])
            con = _constraint_3obj(x[:, 0], x[:, 1], s, idx, (0.5, 0.5, 0.5))
        else:
            xx = x[:, 2:]
            temp = xx.shape[1]
            s = torch.sum(
                (xx - torch.cos(0.25 * torch.pi * temp * (x[:, [0]] + x[:, [1]]) / self.D)) ** 2,
                dim=1,
            )
            f1 = torch.cos(0.5 * torch.pi * x[:, 0]) * torch.cos(0.5 * torch.pi * x[:, 1]) + s
            f2 = torch.cos(0.5 * torch.pi * x[:, 0]) * torch.sin(0.5 * torch.pi * x[:, 1]) + s
            f3 = torch.sin(0.5 * torch.pi * x[:, 0]) + s
            obj = torch.column_stack([f1, f2, f3])
            con = _constraint_3obj(x[:, 0], x[:, 1], s, idx, (0.5, 0.5, 0.5))
        return self._pack_population(x, obj, con)

    def get_optimum(self, n=1000):
        """Generate objective-space reference points for metrics."""
        idx = self.idx
        dtype = torch.float64
        if idx in (1, 4):
            values = _reference_dascmop_1_or_4(n, idx)
            return torch.from_numpy(values).to(device=self.device, dtype=self.dtype)
        if idx == 2:
            x = torch.linspace(0, 1, n, dtype=dtype)
            r = torch.column_stack([x, 1 - torch.sqrt(x)]) + 0.5
            theta = torch.tensor(-0.25 * torch.pi, dtype=dtype)
            while True:
                c = (
                    0.25
                    - ((r[:, 0] - 1) * torch.cos(theta) - (r[:, 1] - 0.5) * torch.sin(theta)) ** 2
                    / 0.3
                    - ((r[:, 0] - 1) * torch.sin(theta) + (r[:, 1] - 0.5) * torch.cos(theta)) ** 2
                    / 1.2
                )
                invalid = c > 0
                if not torch.any(invalid):
                    break
                r[invalid] = (r[invalid] - 0.5) * 1.001 + 0.5
            return r
        if idx in (3, 6):
            return torch.tensor(
                [
                    [0.5000, 1.5000],
                    [0.5010, 1.4762],
                    [0.5020, 1.4710],
                    [0.5030, 1.4688],
                    [0.5040, 1.4681],
                    [0.6502, 1.4652],
                    [0.7002, 1.0541],
                    [0.9044, 0.8986],
                    [1.1066, 0.7729],
                    [1.3008, 0.6114],
                    [1.5000, 0.5000],
                    [0.9069, 0.8951],
                    [1.1126, 0.7727],
                    [0.9129, 0.8950],
                    [1.1151, 0.7690],
                    [0.9153, 0.8914],
                    [1.1175, 0.7653],
                    [1.1200, 0.7616],
                    [0.9213, 0.8913],
                    [1.1260, 0.7613],
                    [1.1285, 0.7576],
                ],
                dtype=dtype,
            )
        if idx == 5:
            x = torch.linspace(0, 1, n, dtype=dtype)
            r = torch.column_stack([x, 1 - torch.sqrt(x)]) + 0.5
            r = r[torch.sin(20 * torch.pi * r[:, 0]) >= -1e-10]
            theta = torch.tensor(-0.25 * torch.pi, dtype=dtype)
            while True:
                c = (
                    0.25
                    - ((r[:, 0] - 1) * torch.cos(theta) - (r[:, 1] - 0.5) * torch.sin(theta)) ** 2
                    / 0.3
                    - ((r[:, 0] - 1) * torch.sin(theta) + (r[:, 1] - 0.5) * torch.cos(theta)) ** 2
                    / 1.2
                )
                invalid = c > 0
                if not torch.any(invalid):
                    break
                r[invalid] = (r[invalid] - 0.5) * 1.001 + 0.5
            return r
        if idx == 7:
            r, _ = uniform_point(n, 3, device=torch.device("cpu"))
            x1 = 1 / (1 + r[:, 1] / torch.clamp(r[:, 0], min=1e-12))
            x2 = r[:, 0] / torch.clamp(x1, min=1e-12)
            c1 = -torch.sin(20 * torch.pi * x1)
            c2 = -torch.cos(20 * torch.pi * x2)
            r = r[~((c1 > 1e-2) | (c2 > 1e-2))]
            return r + 0.5
        if idx in (8, 9):
            r, _ = uniform_point(n, 3, device=torch.device("cpu"))
            r = r / torch.linalg.norm(r, dim=1, keepdim=True).clamp_min(1e-12)
            x2 = torch.atan(r[:, 1] / torch.clamp(r[:, 0], min=1e-12)) / (0.5 * torch.pi)
            x1 = torch.acos(
                torch.clamp(
                    r[:, 0] / torch.clamp(torch.cos(0.5 * torch.pi * x2), min=1e-12), -1.0, 1.0
                )
            ) / (0.5 * torch.pi)
            c1 = -torch.sin(20 * torch.pi * x1)
            c2 = -torch.cos(20 * torch.pi * x2)
            r = r[~((c1 > 1e-2) | (c2 > 1e-2))]
            return r + 0.5
        return torch.empty((0, 3), dtype=dtype)

    def get_pf(self):
        """Generate a drawable Pareto-front representation."""
        dtype = torch.float64
        if self.idx <= 6:
            x, y = torch.meshgrid(
                torch.linspace(0.5, 2.5, 400, dtype=dtype),
                torch.linspace(0.5, 2.5, 400, dtype=dtype),
                indexing="xy",
            )
            if self.idx in (1, 4):
                rad = 1 - 4 * (-x + y - 1)
                x1 = (torch.sqrt(torch.clamp(rad, min=0.0)) - 1) / 2
            elif self.idx in (2, 5):
                rad = 1 - 4 * (-x + y - 1)
                x1 = ((torch.sqrt(torch.clamp(rad, min=0.0)) - 1) / 2) ** 2
            else:
                x1 = _estimate_x1_das36(x, y)
            sum1 = x - x1
            diff = (0.0, 0.5, 0.5) if self.idx <= 2 else (0.5, 0.5, 0.5)
            c = _constraint_2obj(x1.reshape(-1), sum1.reshape(-1), self.idx, diff)
            valid_x1 = (x1 >= 0.0) & (x1 <= 1.0)
            fes = torch.all(c <= 0, dim=1) & valid_x1.reshape(-1)
            z = torch.full_like(x, torch.nan)
            z.reshape(-1)[fes] = 0.0
            return (x, y, z)

        x1, x2 = torch.meshgrid(
            torch.linspace(0.0, 1.0, 100, dtype=dtype),
            torch.linspace(0.0, 1.0, 100, dtype=dtype),
            indexing="xy",
        )
        sum1 = torch.full((x1.numel(),), 0.5, dtype=dtype)
        c = _constraint_3obj(x1.reshape(-1), x2.reshape(-1), sum1, self.idx, (0.5, 0.5, 0.5))
        fes = torch.all(c <= 0, dim=1).reshape_as(x1)
        if self.idx == 7:
            px = x1 * x2
            py = (1 - x1) * x2
            pz = 1 - x2
        else:
            px = torch.cos(0.5 * torch.pi * x1) * torch.cos(0.5 * torch.pi * x2)
            py = torch.cos(0.5 * torch.pi * x1) * torch.sin(0.5 * torch.pi * x2)
            pz = torch.sin(0.5 * torch.pi * x1)
        pz = pz.clone()
        pz[~fes] = torch.nan
        return (px + 0.5, py + 0.5, pz + 0.5)


def _make_dascmop_class(index: int) -> type[DASCMOP]:
    """Create a named suite class bound to one DAS-CMOP problem index."""

    class DASCMOPVariant(DASCMOP):
        """Represent one indexed DAS-CMOP benchmark problem."""

        name = f"DASCMOP{index}"

        def __init__(self, **kwargs) -> None:
            """Initialize the indexed DAS-CMOP problem."""
            super().__init__(index, **kwargs)

    DASCMOPVariant.__name__ = f"DASCMOP{index}"
    DASCMOPVariant.__qualname__ = DASCMOPVariant.__name__
    DASCMOPVariant.__doc__ = f"Implement the DASCMOP{index} benchmark problem."
    return DASCMOPVariant


DAS_CLASSES = {f"DASCMOP{index}": _make_dascmop_class(index) for index in range(1, 10)}
globals().update(DAS_CLASSES)
