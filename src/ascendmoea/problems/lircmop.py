"""Vectorized implementation of the lircmop problem."""

from __future__ import annotations

import torch

from ..core import Problem
from ..utils import uniform_point


def _ellipse_constraints(obj: torch.Tensor, p, q, a, b, r=0.1, theta=-0.25 * torch.pi):
    """Compute ellipse constraints."""
    p = torch.as_tensor(p, dtype=obj.dtype, device=obj.device)
    q = torch.as_tensor(q, dtype=obj.dtype, device=obj.device)
    a = torch.as_tensor(a, dtype=obj.dtype, device=obj.device)
    b = torch.as_tensor(b, dtype=obj.dtype, device=obj.device)
    theta = torch.as_tensor(theta, dtype=obj.dtype, device=obj.device)
    cth = torch.cos(theta)
    sth = torch.sin(theta)
    x = obj[:, [0]]
    y = obj[:, [1]]
    xx = ((x - p) * cth - (y - q) * sth) ** 2 / (a**2)
    yy = ((x - p) * sth + (y - q) * cth) ** 2 / (b**2)
    return r - xx - yy


def _wave_constraint(obj: torch.Tensor, cval: float, alpha=0.25 * torch.pi):
    """Compute wave constraint."""
    alpha = torch.as_tensor(alpha, dtype=obj.dtype, device=obj.device)
    salpha = torch.sin(alpha)
    calpha = torch.cos(alpha)
    return (
        cval
        - obj[:, 0] * salpha
        - obj[:, 1] * calpha
        + torch.sin(4 * torch.pi * (obj[:, 0] * calpha - obj[:, 1] * salpha))
    )


class LIRCMOP(Problem):
    """Implement the LIRCMOP vectorized optimization problem."""

    def __init__(self, idx: int, n=100, m=None, d=30, max_fe=10000):
        """Initialize the instance."""
        self.idx = int(idx)
        self.name = f"LIRCMOP{self.idx}"
        super().__init__(n=n, m=3 if idx >= 13 else 2, d=d, max_fe=max_fe)

    def setting(self):
        """Configure problem dimensions, bounds, encoding, and static data."""
        self.M = 3 if self.idx >= 13 else 2
        self.lower = torch.zeros(self.D, dtype=torch.float64, device=self.device)
        self.upper = torch.ones(self.D, dtype=torch.float64, device=self.device)

    def _sum_sincos(self, x: torch.Tensor):
        """Compute sum sincos."""
        j = torch.arange(2, self.D + 1, device=x.device)
        j_idx = j - 1
        coef = (0.5 * j.to(x.dtype) / self.D * torch.pi).unsqueeze(0)
        x1 = x[:, [0]]
        term = x[:, j_idx]
        odd = j % 2 == 1
        even = ~odd
        s1 = torch.sum((term[:, odd] - torch.sin(coef[:, odd] * x1)) ** 2, dim=1)
        s2 = torch.sum((term[:, even] - torch.cos(coef[:, even] * x1)) ** 2, dim=1)
        return s1, s2

    def evaluation(self, dec):
        """Repair and evaluate a complete decision batch."""
        x = self._prepare_dec(dec)
        if x.shape[0] == 0:
            return self._empty_population()
        i = self.idx
        if i <= 4:
            odd = x[:, 2::2]
            even = x[:, 1::2]
            if i == 1:
                g1 = torch.sum((odd - torch.sin(0.5 * torch.pi * x[:, [0]])) ** 2, dim=1)
                g2 = torch.sum((even - torch.cos(0.5 * torch.pi * x[:, [0]])) ** 2, dim=1)
            else:
                g1 = torch.sum((odd - x[:, [0]]) ** 2, dim=1)
                g2 = torch.sum((even - x[:, [0]]) ** 2, dim=1)
            f1 = x[:, 0] + g1
            f2 = (1 - x[:, 0] ** 2 + g2) if i in (1, 3) else (1 - torch.sqrt(x[:, 0]) + g2)
            c = [(0.5 - g1) * (0.51 - g1), (0.5 - g2) * (0.51 - g2)]
            if i in (3, 4):
                c.append(0.5 - torch.sin(20 * torch.pi * x[:, 0]))
            con = torch.column_stack(c)
            obj = torch.column_stack([f1, f2])
        elif i <= 8:
            s1, s2 = self._sum_sincos(x)
            gx = 0.7057
            f1 = x[:, 0] + 10 * s1 + gx
            f2 = (
                (1 - torch.sqrt(x[:, 0]) + 10 * s2 + gx)
                if i in (5, 7)
                else (1 - x[:, 0] ** 2 + 10 * s2 + gx)
            )
            obj = torch.column_stack([f1, f2])
            if i == 5:
                con = _ellipse_constraints(obj, [1.6, 2.5], [1.6, 2.5], [2, 2], [4, 8])
            elif i == 6:
                con = _ellipse_constraints(obj, [1.8, 2.8], [1.8, 2.8], [2, 2], [8, 8])
            else:
                con = _ellipse_constraints(
                    obj, [1.2, 2.25, 3.5], [1.2, 2.25, 3.5], [2, 2.5, 2.5], [6, 12, 10]
                )
        elif i <= 12:
            s1, s2 = self._sum_sincos(x)
            f1 = 1.7057 * x[:, 0] * (10 * s1 + 1)
            f2 = (
                1.7057
                * ((1 - torch.sqrt(x[:, 0])) if i in (10, 11) else (1 - x[:, 0] ** 2))
                * (10 * s2 + 1)
            )
            obj = torch.column_stack([f1, f2])
            if i == 9:
                e = _ellipse_constraints(obj, [1.4], [1.4], [1.5], [6])[:, 0]
                w = _wave_constraint(obj, 2.0)
            elif i == 10:
                e = _ellipse_constraints(obj, [1.1], [1.2], [2.0], [4.0])[:, 0]
                w = _wave_constraint(obj, 1.0)
            elif i == 11:
                e = _ellipse_constraints(obj, [1.2], [1.2], [1.5], [5.0])[:, 0]
                w = _wave_constraint(obj, 2.1)
            else:
                e = _ellipse_constraints(obj, [1.6], [1.6], [1.5], [6.0])[:, 0]
                w = _wave_constraint(obj, 2.5)
            con = torch.column_stack([e, w])
        else:
            sum1 = torch.sum(10 * (x[:, 2:] - 0.5) ** 2, dim=1)
            scale = 1.7057 + sum1
            f1 = scale * torch.cos(0.5 * torch.pi * x[:, 0]) * torch.cos(0.5 * torch.pi * x[:, 1])
            f2 = scale * torch.cos(0.5 * torch.pi * x[:, 0]) * torch.sin(0.5 * torch.pi * x[:, 1])
            f3 = scale * torch.sin(0.5 * torch.pi * x[:, 0])
            obj = torch.column_stack([f1, f2, f3])
            gx = torch.sum(obj**2, dim=1)
            c1 = (gx - 9) * (4 - gx)
            c2 = (gx - 3.61) * (3.24 - gx)
            if i == 13:
                con = torch.column_stack([c1, c2])
            else:
                c3 = (gx - 3.0625) * (2.56 - gx)
                con = torch.column_stack([c1, c2, c3])
        return self._pack_population(x, obj, con)

    def get_optimum(self, n=1000):
        """Generate objective-space reference points for metrics."""
        i = self.idx
        if i <= 4:
            x = torch.linspace(0, 1, n, dtype=torch.float64)
            y = 1 - x**2 if i in (1, 3) else 1 - torch.sqrt(x)
            if i in (3, 4):
                keep = torch.sin(20 * torch.pi * x) >= 0.5
                x, y = x[keep], y[keep]
            return torch.column_stack([x + 0.5, y + 0.5])
        if i == 5:
            x = torch.linspace(0, 1, n, dtype=torch.float64)
            r = torch.column_stack([x, 1 - torch.sqrt(x)]) + 0.7057
            c = _ellipse_constraints(r, [1.6, 2.5], [1.6, 2.5], [2, 2], [4, 8])
            return r[torch.all(c <= 0, dim=1)]
        if i == 6:
            x = torch.linspace(0, 1, n, dtype=torch.float64)
            r = torch.column_stack([x, 1 - x**2]) + 0.7057
            c = _ellipse_constraints(r, [1.8, 2.8], [1.8, 2.8], [2, 2], [8, 8])
            return r[torch.all(c <= 0, dim=1)]
        if i in (7, 8):
            x = torch.linspace(0, 1, n, dtype=torch.float64)
            r = torch.column_stack([x, 1 - torch.sqrt(x)]) + 0.7057
            theta = torch.tensor(-0.25 * torch.pi, dtype=torch.float64)
            while True:
                c1 = (
                    0.1
                    - ((r[:, 0] - 1.2) * torch.cos(theta) - (r[:, 1] - 1.2) * torch.sin(theta)) ** 2
                    / (2**2)
                    - ((r[:, 0] - 1.2) * torch.sin(theta) + (r[:, 1] - 1.2) * torch.cos(theta)) ** 2
                    / (6**2)
                )
                invalid = c1 > 0
                if not torch.any(invalid):
                    break
                r[invalid] = (r[invalid] - 0.7057) * 1.001 + 0.7057
            return r
        if i == 9:
            x = torch.linspace(0, 1, n, dtype=torch.float64)
            r = torch.column_stack([x, 1 - x**2]) * 1.7057
            e = _ellipse_constraints(r, [1.4], [1.4], [1.5], [6])[:, 0]
            w = _wave_constraint(r, 2.0)
            r = r[(e <= 0) & (w <= 0)]
            return torch.cat(
                [r, torch.tensor([[0.0, 2.1820], [1.8560, 0.0]], dtype=torch.float64)], dim=0
            )
        if i == 10:
            x = torch.linspace(0, 1, n, dtype=torch.float64)
            r = torch.column_stack([x, 1 - torch.sqrt(x)]) * 1.7057
            e = _ellipse_constraints(r, [1.1], [1.2], [2.0], [4.0])[:, 0]
            w = _wave_constraint(r, 1.0)
            r = r[(e <= 0) & (w <= 0)]
            return torch.cat([r, torch.tensor([[1.7470, 0.0]], dtype=torch.float64)], dim=0)
        if i == 11:
            return torch.tensor(
                [
                    [1.3965, 0.1591],
                    [1.0430, 0.5127],
                    [0.6894, 0.8662],
                    [0.3359, 1.2198],
                    [0.0106, 1.6016],
                    [0.0, 2.1910],
                    [1.8730, 0.0],
                ],
                dtype=torch.float64,
            )
        if i == 12:
            return torch.tensor(
                [
                    [1.6794, 0.4419],
                    [1.3258, 0.7955],
                    [0.9723, 1.1490],
                    [2.0320, 0.0990],
                    [0.6187, 1.5026],
                    [0.2652, 1.8562],
                    [0.0, 2.2580],
                    [2.5690, 0.0],
                ],
                dtype=torch.float64,
            )
        w, _ = uniform_point(n, 3, device=torch.device("cpu"))
        scale = 1.7057 if i == 13 else torch.sqrt(torch.tensor(3.0625, dtype=torch.float64))
        return scale * w / torch.linalg.norm(w, dim=1, keepdim=True).clamp_min(1e-12)

    def get_pf(self):
        """Generate a drawable Pareto-front representation."""
        i = self.idx
        dtype = torch.float64
        if i in (1, 2):
            return self.get_optimum(100)
        if i == 3:
            x = torch.linspace(0, 1, 100, dtype=dtype)
            y = 1 - x**2
            y = y.clone()
            y[torch.sin(20 * torch.pi * x) < 0.5] = torch.nan
            return torch.column_stack([x, y]) + 0.5
        if i == 4:
            x = torch.linspace(0, 1, 100, dtype=dtype)
            y = 1 - torch.sqrt(x)
            y = y.clone()
            y[torch.sin(20 * torch.pi * x) < 0.5] = torch.nan
            return torch.column_stack([x, y]) + 0.5
        if i <= 8:
            x, y = torch.meshgrid(
                torch.linspace(0.7057, 5.0, 400, dtype=dtype),
                torch.linspace(0.7057, 5.0, 400, dtype=dtype),
                indexing="xy",
            )
            obj = torch.column_stack([x.reshape(-1), y.reshape(-1)])
            if i == 5:
                c = _ellipse_constraints(obj, [1.6, 2.5], [1.6, 2.5], [2, 2], [4, 8])
            elif i == 6:
                c = _ellipse_constraints(obj, [1.8, 2.8], [1.8, 2.8], [2, 2], [8, 8])
            else:
                c = _ellipse_constraints(
                    obj, [1.2, 2.25, 3.5], [1.2, 2.25, 3.5], [2, 2.5, 2.5], [6, 12, 10]
                )
            fes = torch.all(c <= 0, dim=1).reshape_as(x)
            if i == 6:
                cond = (x - 0.7057) ** 2 + y >= 1.7057
            else:
                cond = torch.sqrt(torch.clamp(x - 0.7057, min=0.0)) + y >= 1.7057
            z = torch.full_like(x, torch.nan)
            z[fes & cond] = 0.0
            return (x, y, z)
        if i <= 12:
            x, y = torch.meshgrid(
                torch.linspace(0.0, 5.0, 400, dtype=dtype),
                torch.linspace(0.0, 5.0, 400, dtype=dtype),
                indexing="xy",
            )
            obj = torch.column_stack([x.reshape(-1), y.reshape(-1)])
            if i == 9:
                e = _ellipse_constraints(obj, [1.4], [1.4], [1.5], [6])[:, 0]
                w = _wave_constraint(obj, 2.0)
                cond = (x / 1.7057) ** 2 + y / 1.7057 >= 1
            elif i == 10:
                e = _ellipse_constraints(obj, [1.1], [1.2], [2.0], [4.0])[:, 0]
                w = _wave_constraint(obj, 1.0)
                cond = torch.sqrt(torch.clamp(x / 1.7057, min=0.0)) + y / 1.7057 >= 1
            elif i == 11:
                e = _ellipse_constraints(obj, [1.2], [1.2], [1.5], [5.0])[:, 0]
                w = _wave_constraint(obj, 2.1)
                cond = torch.sqrt(torch.clamp(x / 1.7057, min=0.0)) + y / 1.7057 >= 1
            else:
                e = _ellipse_constraints(obj, [1.6], [1.6], [1.5], [6.0])[:, 0]
                w = _wave_constraint(obj, 2.5)
                cond = (x / 1.7057) ** 2 + y / 1.7057 >= 1
            fes = (e <= 0) & (w <= 0)
            z = torch.full_like(x, torch.nan)
            z.reshape(-1)[fes] = 0.0
            z[~cond] = torch.nan
            return (x, y, z)
        a = torch.linspace(0, torch.pi / 2, 10, dtype=dtype)
        sa = torch.sin(a)
        ca = torch.cos(a)
        scale = 1.7057 if i == 13 else torch.sqrt(torch.tensor(3.0625, dtype=dtype))
        x = sa[:, None] * ca[None, :] * scale
        y = sa[:, None] * sa[None, :] * scale
        z = ca[:, None] * torch.ones_like(ca[None, :]) * scale
        return (x, y, z)


def _make_lircmop_class(index: int) -> type[LIRCMOP]:
    """Create a named suite class bound to one LIR-CMOP problem index."""

    class LIRCMOPVariant(LIRCMOP):
        """Represent one indexed LIR-CMOP benchmark problem."""

        name = f"LIRCMOP{index}"

        def __init__(self, **kwargs) -> None:
            """Initialize the indexed LIR-CMOP problem."""
            super().__init__(index, **kwargs)

    LIRCMOPVariant.__name__ = f"LIRCMOP{index}"
    LIRCMOPVariant.__qualname__ = LIRCMOPVariant.__name__
    LIRCMOPVariant.__doc__ = f"Implement the LIRCMOP{index} benchmark problem."
    return LIRCMOPVariant


LIR_CLASSES = {f"LIRCMOP{index}": _make_lircmop_class(index) for index in range(1, 15)}
globals().update(LIR_CLASSES)
