"""Vectorized implementation of the cf problem."""

from __future__ import annotations

import numpy as np
import torch

from ..core import Problem
from ..utils import uniform_point


class CF1(Problem):
    """Define the CF1 benchmark with batched tensor evaluation."""

    name = "CF1"

    def setting(self):
        """Configure problem dimensions, bounds, encoding, and static data."""
        self.M = 2
        if self.D is None:
            self.D = 10
        self.lower = torch.zeros(self.D, dtype=torch.float64, device=self.device)
        self.upper = torch.ones(self.D, dtype=torch.float64, device=self.device)

    def evaluation(self, dec):
        """Repair and evaluate a complete decision batch."""
        x = self._prepare_dec(dec)
        if x.shape[0] == 0:
            return self._empty_population()
        d = x.shape[1]
        j1 = torch.arange(2, d, 2, device=x.device)
        j2 = torch.arange(1, d, 2, device=x.device)
        a1 = 0.5 * (1 + 3 * ((j1 + 1).to(x.dtype) - 2) / (d - 2))
        a2 = 0.5 * (1 + 3 * ((j2 + 1).to(x.dtype) - 2) / (d - 2))
        f1 = x[:, 0] + 2 * torch.mean((x[:, j1] - x[:, [0]] ** a1.unsqueeze(0)) ** 2, dim=1)
        f2 = 1 - x[:, 0] + 2 * torch.mean((x[:, j2] - x[:, [0]] ** a2.unsqueeze(0)) ** 2, dim=1)
        c = 1 - f1 - f2 + torch.abs(torch.sin(10 * torch.pi * (f1 - f2 + 1)))
        return self._pack_population(x, torch.column_stack([f1, f2]), c.unsqueeze(1))

    def get_optimum(self, n=1000):
        """Generate objective-space reference points for metrics."""
        x = torch.linspace(0, 1, 21, dtype=torch.float64)
        return torch.column_stack([x, 1 - x])

    def get_pf(self):
        """Generate a drawable Pareto-front representation."""
        return None


class CF2(Problem):
    """Define the CF2 benchmark with batched tensor evaluation."""

    name = "CF2"

    def setting(self):
        """Configure problem dimensions, bounds, encoding, and static data."""
        self.M = 2
        if self.D is None:
            self.D = 10
        self.lower = torch.cat(
            [
                torch.tensor([0.0], dtype=torch.float64, device=self.device),
                -torch.ones(self.D - 1, dtype=torch.float64, device=self.device),
            ]
        )
        self.upper = torch.ones(self.D, dtype=torch.float64, device=self.device)

    def evaluation(self, dec):
        """Repair and evaluate a complete decision batch."""
        x = self._prepare_dec(dec)
        if x.shape[0] == 0:
            return self._empty_population()
        d = x.shape[1]
        j1 = torch.arange(2, d, 2, device=x.device)
        j2 = torch.arange(1, d, 2, device=x.device)
        jd = torch.arange(1, d + 1, device=x.device, dtype=x.dtype).unsqueeze(0)
        f1 = x[:, 0] + 2 * torch.mean(
            (x[:, j1] - torch.sin(6 * torch.pi * x[:, [0]] + jd[:, j1] * torch.pi / d)) ** 2, dim=1
        )
        f2 = (
            1
            - torch.sqrt(x[:, 0])
            + 2
            * torch.mean(
                (x[:, j2] - torch.cos(6 * torch.pi * x[:, [0]] + jd[:, j2] * torch.pi / d)) ** 2,
                dim=1,
            )
        )
        root_f1 = torch.sqrt(f1)
        t = f2 + root_f1 - torch.sin(2 * torch.pi * (root_f1 - f2 + 1)) - 1
        c = -t / (1 + torch.exp(4 * torch.abs(t)))
        return self._pack_population(x, torch.column_stack([f1, f2]), c.unsqueeze(1))

    def get_optimum(self, n=1000):
        """Generate objective-space reference points for metrics."""
        x = torch.linspace(0, 1, n, dtype=torch.float64)
        r = torch.column_stack([x, 1 - torch.sqrt(x)])
        mask = ~(((0 < r[:, 0]) & (r[:, 0] < 1 / 16)) | ((1 / 4 < r[:, 0]) & (r[:, 0] < 9 / 16)))
        return r[mask]

    def get_pf(self):
        """Generate a drawable Pareto-front representation."""
        x = torch.linspace(0, 1, 100, dtype=torch.float64)
        r = torch.column_stack([x, 1 - torch.sqrt(x)])
        invalid = ((0 < r[:, 0]) & (r[:, 0] < 1 / 16)) | ((1 / 4 < r[:, 0]) & (r[:, 0] < 9 / 16))
        r[invalid, :] = torch.nan
        return r


class CF3(Problem):
    """Define the CF3 benchmark with batched tensor evaluation."""

    name = "CF3"

    def setting(self):
        """Configure problem dimensions, bounds, encoding, and static data."""
        self.M = 2
        if self.D is None:
            self.D = 10
        self.lower = torch.cat(
            [
                torch.tensor([0.0], dtype=torch.float64, device=self.device),
                -2 * torch.ones(self.D - 1, dtype=torch.float64, device=self.device),
            ]
        )
        self.upper = torch.cat(
            [
                torch.tensor([1.0], dtype=torch.float64, device=self.device),
                2 * torch.ones(self.D - 1, dtype=torch.float64, device=self.device),
            ]
        )

    def evaluation(self, dec):
        """Repair and evaluate a complete decision batch."""
        x = self._prepare_dec(dec)
        if x.shape[0] == 0:
            return self._empty_population()
        d = x.shape[1]
        j1 = torch.arange(2, d, 2, device=x.device)
        j2 = torch.arange(1, d, 2, device=x.device)
        jd = torch.arange(1, d + 1, device=x.device, dtype=x.dtype).unsqueeze(0)
        y = x - torch.sin(6 * torch.pi * x[:, [0]] + jd * torch.pi / d)
        f1 = x[:, 0] + 2 / len(j1) * (
            4 * torch.sum(y[:, j1] ** 2, dim=1)
            - 2 * torch.prod(torch.cos(20 * y[:, j1] * torch.pi / torch.sqrt(jd[:, j1])), dim=1)
            + 2
        )
        f2 = (
            1
            - x[:, 0] ** 2
            + 2
            / len(j2)
            * (
                4 * torch.sum(y[:, j2] ** 2, dim=1)
                - 2 * torch.prod(torch.cos(20 * y[:, j2] * torch.pi / torch.sqrt(jd[:, j2])), dim=1)
                + 2
            )
        )
        c = 1 - f2 - f1**2 + torch.sin(2 * torch.pi * (f1**2 - f2 + 1))
        return self._pack_population(x, torch.column_stack([f1, f2]), c.unsqueeze(1))

    def get_optimum(self, n=1000):
        """Generate objective-space reference points for metrics."""
        x = torch.linspace(0, 1, n, dtype=torch.float64)
        r = torch.column_stack([x, 1 - x**2])
        mask = ~(
            ((0 < r[:, 0]) & (r[:, 0] < 1 / 2))
            | (
                (torch.sqrt(torch.tensor(1 / 2.0, dtype=torch.float64)) < r[:, 0])
                & (r[:, 0] < torch.sqrt(torch.tensor(3 / 4.0, dtype=torch.float64)))
            )
        )
        return r[mask]

    def get_pf(self):
        """Generate a drawable Pareto-front representation."""
        x = torch.linspace(0, 1, 100, dtype=torch.float64)
        r = torch.column_stack([x, 1 - x**2])
        invalid = ((0 < r[:, 0]) & (r[:, 0] < 1 / 2)) | (
            (torch.sqrt(torch.tensor(1 / 2.0, dtype=torch.float64)) < r[:, 0])
            & (r[:, 0] < torch.sqrt(torch.tensor(3 / 4.0, dtype=torch.float64)))
        )
        r[invalid, :] = torch.nan
        return r


class _CF4to7Base(Problem):
    """Define the _CF4to7Base benchmark with batched tensor evaluation."""

    def setting(self):
        """Configure problem dimensions, bounds, encoding, and static data."""
        self.M = 2
        if self.D is None:
            self.D = 10
        self.lower = torch.cat(
            [
                torch.tensor([0.0], dtype=torch.float64, device=self.device),
                -2 * torch.ones(self.D - 1, dtype=torch.float64, device=self.device),
            ]
        )
        self.upper = torch.cat(
            [
                torch.tensor([1.0], dtype=torch.float64, device=self.device),
                2 * torch.ones(self.D - 1, dtype=torch.float64, device=self.device),
            ]
        )

    def get_optimum(self, n=1000):
        """Generate objective-space reference points for metrics."""
        x = torch.linspace(0, 1, n, dtype=torch.float64)
        y = (1 - x) if self.name in {"CF4", "CF5"} else (1 - x) ** 2
        t1 = (0.5 < x) & (x <= 0.75)
        t2 = 0.75 < x
        if self.name in {"CF4", "CF5"}:
            y[t1] = -0.5 * x[t1] + 0.75
            y[t2] = 1 - x[t2] + 0.125
        else:
            y[t1] = 0.5 * (1 - x[t1])
            y[t2] = 0.25 * torch.sqrt(1 - x[t2])
        return torch.column_stack([x, y])

    def get_pf(self):
        """Generate a drawable Pareto-front representation."""
        return self.get_optimum(100)


class CF4(_CF4to7Base):
    """Define the CF4 benchmark with batched tensor evaluation."""

    name = "CF4"

    def cal_obj(self, x: torch.Tensor):
        """Evaluate objective values for a batch of decision vectors."""
        d = x.shape[1]
        j1 = torch.arange(2, d, 2, device=x.device)
        j2 = torch.arange(1, d, 2, device=x.device)
        jd = torch.arange(1, d + 1, device=x.device, dtype=x.dtype).unsqueeze(0)
        y = x - torch.sin(6 * torch.pi * x[:, [0]] + jd * torch.pi / d)
        h = y**2
        temp = y[:, 1] < 1.5 * (1 - torch.sqrt(torch.tensor(0.5, dtype=x.dtype, device=x.device)))
        h[temp, 1] = torch.abs(y[temp, 1])
        h[~temp, 1] = 0.125 + (y[~temp, 1] - 1) ** 2
        f1 = x[:, 0] + torch.sum(h[:, j1], dim=1)
        f2 = 1 - x[:, 0] + torch.sum(h[:, j2], dim=1)
        return torch.column_stack([f1, f2])

    def cal_con(self, x: torch.Tensor):
        """Evaluate inequality constraints for a batch of decision vectors."""
        t = (
            x[:, 1]
            - torch.sin(6 * torch.pi * x[:, 0] + 2 * torch.pi / x.shape[1])
            - 0.5 * x[:, 0]
            + 0.25
        )
        return (-t / (1 + torch.exp(4 * torch.abs(t)))).unsqueeze(1)


class CF5(_CF4to7Base):
    """Define the CF5 benchmark with batched tensor evaluation."""

    name = "CF5"

    def cal_obj(self, x: torch.Tensor):
        """Evaluate objective values for a batch of decision vectors."""
        d = x.shape[1]
        j1 = torch.arange(2, d, 2, device=x.device)
        j2 = torch.arange(1, d, 2, device=x.device)
        jd = torch.arange(1, d + 1, device=x.device, dtype=x.dtype).unsqueeze(0)
        y = torch.zeros_like(x)
        y[:, j1] = x[:, j1] - 0.8 * x[:, [0]] * torch.cos(
            6 * torch.pi * x[:, [0]] + jd[:, j1] * torch.pi / d
        )
        y[:, j2] = x[:, j2] - 0.8 * x[:, [0]] * torch.sin(
            6 * torch.pi * x[:, [0]] + jd[:, j2] * torch.pi / d
        )
        h = 2 * y**2 - torch.cos(4 * torch.pi * y) + 1
        temp = y[:, 1] < 1.5 * (1 - torch.sqrt(torch.tensor(0.5, dtype=x.dtype, device=x.device)))
        h[temp, 1] = torch.abs(y[temp, 1])
        h[~temp, 1] = 0.125 + (y[~temp, 1] - 1) ** 2
        f1 = x[:, 0] + torch.sum(h[:, j1], dim=1)
        f2 = 1 - x[:, 0] + torch.sum(h[:, j2], dim=1)
        return torch.column_stack([f1, f2])

    def cal_con(self, x: torch.Tensor):
        """Evaluate inequality constraints for a batch of decision vectors."""
        c = (
            -x[:, 1]
            + 0.8 * x[:, 0] * torch.sin(6 * torch.pi * x[:, 0] + 2 * torch.pi / x.shape[1])
            + 0.5 * x[:, 0]
            - 0.25
        )
        return c.unsqueeze(1)


class CF6(_CF4to7Base):
    """Define the CF6 benchmark with batched tensor evaluation."""

    name = "CF6"

    def cal_obj(self, x: torch.Tensor):
        """Evaluate objective values for a batch of decision vectors."""
        d = x.shape[1]
        j1 = torch.arange(2, d, 2, device=x.device)
        j2 = torch.arange(1, d, 2, device=x.device)
        jd = torch.arange(1, d + 1, device=x.device, dtype=x.dtype).unsqueeze(0)
        y = torch.zeros_like(x)
        y[:, j1] = x[:, j1] - 0.8 * x[:, [0]] * torch.cos(
            6 * torch.pi * x[:, [0]] + jd[:, j1] * torch.pi / d
        )
        y[:, j2] = x[:, j2] - 0.8 * x[:, [0]] * torch.sin(
            6 * torch.pi * x[:, [0]] + jd[:, j2] * torch.pi / d
        )
        f1 = x[:, 0] + torch.sum(y[:, j1] ** 2, dim=1)
        f2 = (1 - x[:, 0]) ** 2 + torch.sum(y[:, j2] ** 2, dim=1)
        return torch.column_stack([f1, f2])

    def cal_con(self, x: torch.Tensor):
        """Evaluate inequality constraints for a batch of decision vectors."""
        t1 = 0.5 * (1 - x[:, 0]) - (1 - x[:, 0]) ** 2
        t2 = 0.25 * torch.sqrt(1 - x[:, 0]) - 0.5 * (1 - x[:, 0])
        c1 = (
            -x[:, 1]
            + 0.8 * x[:, 0] * torch.sin(6 * torch.pi * x[:, 0] + 2 * torch.pi / x.shape[1])
            + torch.sign(t1) * torch.sqrt(torch.abs(t1))
        )
        c2 = (
            -x[:, 3]
            + 0.8 * x[:, 0] * torch.sin(6 * torch.pi * x[:, 0] + 4 * torch.pi / x.shape[1])
            + torch.sign(t2) * torch.sqrt(torch.abs(t2))
        )
        return torch.column_stack([c1, c2])


class CF7(_CF4to7Base):
    """Define the CF7 benchmark with batched tensor evaluation."""

    name = "CF7"

    def cal_obj(self, x: torch.Tensor):
        """Evaluate objective values for a batch of decision vectors."""
        d = x.shape[1]
        j1 = torch.arange(2, d, 2, device=x.device)
        j2 = torch.arange(1, d, 2, device=x.device)
        jd = torch.arange(1, d + 1, device=x.device, dtype=x.dtype).unsqueeze(0)
        y = torch.zeros_like(x)
        y[:, j1] = x[:, j1] - torch.cos(6 * torch.pi * x[:, [0]] + jd[:, j1] * torch.pi / d)
        y[:, j2] = x[:, j2] - torch.sin(6 * torch.pi * x[:, [0]] + jd[:, j2] * torch.pi / d)
        h = 2 * y**2 - torch.cos(4 * torch.pi * y) + 1
        h[:, [1, 3]] = y[:, [1, 3]] ** 2
        f1 = x[:, 0] + torch.sum(h[:, j1], dim=1)
        f2 = (1 - x[:, 0]) ** 2 + torch.sum(h[:, j2], dim=1)
        return torch.column_stack([f1, f2])

    def cal_con(self, x: torch.Tensor):
        """Evaluate inequality constraints for a batch of decision vectors."""
        t1 = 0.5 * (1 - x[:, 0]) - (1 - x[:, 0]) ** 2
        t2 = 0.25 * torch.sqrt(1 - x[:, 0]) - 0.5 * (1 - x[:, 0])
        c1 = (
            -x[:, 1]
            + torch.sin(6 * torch.pi * x[:, 0] + 2 * torch.pi / x.shape[1])
            + torch.sign(t1) * torch.sqrt(torch.abs(t1))
        )
        c2 = (
            -x[:, 3]
            + torch.sin(6 * torch.pi * x[:, 0] + 4 * torch.pi / x.shape[1])
            + torch.sign(t2) * torch.sqrt(torch.abs(t2))
        )
        return torch.column_stack([c1, c2])


class CF8(Problem):
    """Define the CF8 benchmark with batched tensor evaluation."""

    name = "CF8"

    def setting(self):
        """Configure problem dimensions, bounds, encoding, and static data."""
        self.M = 3
        if self.D is None:
            self.D = 10
        self.lower = torch.cat(
            [
                torch.zeros(2, dtype=torch.float64, device=self.device),
                -4 * torch.ones(self.D - 2, dtype=torch.float64, device=self.device),
            ]
        )
        self.upper = torch.cat(
            [
                torch.ones(2, dtype=torch.float64, device=self.device),
                4 * torch.ones(self.D - 2, dtype=torch.float64, device=self.device),
            ]
        )

    def evaluation(self, dec):
        """Repair and evaluate a complete decision batch."""
        x = self._prepare_dec(dec)
        if x.shape[0] == 0:
            return self._empty_population()
        d = x.shape[1]
        j1 = torch.arange(3, d, 3, device=x.device)
        j2 = torch.arange(4, d, 3, device=x.device)
        j3 = torch.arange(2, d, 3, device=x.device)
        jd = torch.arange(1, d + 1, device=x.device, dtype=x.dtype).unsqueeze(0)
        y = x - 2 * x[:, [1]] * torch.sin(2 * torch.pi * x[:, [0]] + jd * torch.pi / d)
        f1 = torch.cos(0.5 * x[:, 0] * torch.pi) * torch.cos(
            0.5 * x[:, 1] * torch.pi
        ) + 2 * torch.mean(y[:, j1] ** 2, dim=1)
        f2 = torch.cos(0.5 * x[:, 0] * torch.pi) * torch.sin(
            0.5 * x[:, 1] * torch.pi
        ) + 2 * torch.mean(y[:, j2] ** 2, dim=1)
        f3 = torch.sin(0.5 * x[:, 0] * torch.pi) + 2 * torch.mean(y[:, j3] ** 2, dim=1)
        den = 1 - f3**2
        c = (
            1
            - (f1**2 + f2**2) / den
            + 4 * torch.abs(torch.sin(2 * torch.pi * ((f1**2 - f2**2) / den + 1)))
        )
        return self._pack_population(x, torch.column_stack([f1, f2, f3]), c.unsqueeze(1))

    def get_optimum(self, n=1000):
        """Generate objective-space reference points for metrics."""
        n = int(torch.ceil(torch.tensor(n / 5)).item() * 5)
        r = torch.zeros((n, 3), dtype=torch.float64)
        z = torch.sin(torch.linspace(0, 1, n // 5, dtype=torch.float64) * torch.pi / 2)
        r[:, 2] = z.repeat(5)
        for i in range(5):
            s = slice(i * (n // 5), (i + 1) * (n // 5))
            r[s, 0] = torch.sqrt(torch.tensor(i / 4, dtype=torch.float64) * (1 - r[s, 2] ** 2))
        r[:, 1] = torch.sqrt(torch.clamp(1 - r[:, 0] ** 2 - r[:, 2] ** 2, min=0))
        return r

    def get_pf(self):
        """Generate a drawable Pareto-front representation."""
        return None


class _CF9_10_Base(Problem):
    """Define the _CF9_10_Base benchmark with batched tensor evaluation."""

    name = ""

    def setting(self):
        """Configure problem dimensions, bounds, encoding, and static data."""
        self.M = 3
        if self.D is None:
            self.D = 10
        self.lower = torch.cat(
            [
                torch.zeros(2, dtype=torch.float64, device=self.device),
                -2 * torch.ones(self.D - 2, dtype=torch.float64, device=self.device),
            ]
        )
        self.upper = torch.cat(
            [
                torch.ones(2, dtype=torch.float64, device=self.device),
                2 * torch.ones(self.D - 2, dtype=torch.float64, device=self.device),
            ]
        )

    def _base_obj(self, x: torch.Tensor, harmonic=False):
        """Evaluate the shared unconstrained objective terms for CF8 through CF10."""
        d = x.shape[1]
        j1 = torch.arange(3, d, 3, device=x.device)
        j2 = torch.arange(4, d, 3, device=x.device)
        j3 = torch.arange(2, d, 3, device=x.device)
        jd = torch.arange(1, d + 1, device=x.device, dtype=x.dtype).unsqueeze(0)
        y = x - 2 * x[:, [1]] * torch.sin(2 * torch.pi * x[:, [0]] + jd * torch.pi / d)
        if harmonic:
            g1 = 4 * y[:, j1] ** 2 - torch.cos(8 * torch.pi * y[:, j1]) + 1
            g2 = 4 * y[:, j2] ** 2 - torch.cos(8 * torch.pi * y[:, j2]) + 1
            g3 = 4 * y[:, j3] ** 2 - torch.cos(8 * torch.pi * y[:, j3]) + 1
        else:
            g1 = y[:, j1] ** 2
            g2 = y[:, j2] ** 2
            g3 = y[:, j3] ** 2
        f1 = torch.cos(0.5 * x[:, 0] * torch.pi) * torch.cos(
            0.5 * x[:, 1] * torch.pi
        ) + 2 * torch.mean(g1, dim=1)
        f2 = torch.cos(0.5 * x[:, 0] * torch.pi) * torch.sin(
            0.5 * x[:, 1] * torch.pi
        ) + 2 * torch.mean(g2, dim=1)
        f3 = torch.sin(0.5 * x[:, 0] * torch.pi) + 2 * torch.mean(g3, dim=1)
        den = 1 - f3**2
        return f1, f2, f3, den

    def get_optimum(self, n=1000):
        """Generate objective-space reference points for metrics."""
        r, _ = uniform_point(n, 3, device=torch.device("cpu"))
        values = r.detach().cpu().numpy().astype(np.float64, copy=False)
        values = values / np.sqrt(np.sum(values**2, axis=1, keepdims=True))
        x = values[:, 0]
        z = values[:, 2]
        invalid = ((1e-5 < x) & (x < np.sqrt((1 - z**2) / 4))) | (
            (np.sqrt((1 - z**2) / 2) < x) & (x < np.sqrt(3 * (1 - z**2) / 4))
        )
        return torch.from_numpy(values[~invalid]).to(device=self.device, dtype=self.dtype)

    def get_pf(self):
        """Generate a drawable Pareto-front representation."""
        a = torch.linspace(0, torch.pi / 2, 20, dtype=torch.float64)
        sa = torch.sin(a)
        ca = torch.cos(a)
        x = sa[:, None] * ca[None, :]
        y = sa[:, None] * sa[None, :]
        z = ca[:, None] * torch.ones_like(ca[None, :])
        invalid = ((1e-5 < x) & (x < torch.sqrt(torch.clamp((1 - z**2) / 4, min=0.0)))) | (
            (torch.sqrt(torch.clamp((1 - z**2) / 2, min=0.0)) < x)
            & (x < torch.sqrt(torch.clamp(3 * (1 - z**2) / 4, min=0.0)))
        )
        z = z.clone()
        z[invalid] = torch.nan
        return (x, y, z)


class CF9(_CF9_10_Base):
    """Define the CF9 benchmark with batched tensor evaluation."""

    name = "CF9"

    def evaluation(self, dec):
        """Repair and evaluate a complete decision batch."""
        x = self._prepare_dec(dec)
        if x.shape[0] == 0:
            return self._empty_population()
        f1, f2, f3, den = self._base_obj(x, harmonic=False)
        c = 1 - (f1**2 + f2**2) / den + 3 * torch.sin(2 * torch.pi * ((f1**2 - f2**2) / den + 1))
        return self._pack_population(x, torch.column_stack([f1, f2, f3]), c.unsqueeze(1))


class CF10(_CF9_10_Base):
    """Define the CF10 benchmark with batched tensor evaluation."""

    name = "CF10"

    def evaluation(self, dec):
        """Repair and evaluate a complete decision batch."""
        x = self._prepare_dec(dec)
        if x.shape[0] == 0:
            return self._empty_population()
        f1, f2, f3, den = self._base_obj(x, harmonic=True)
        c = 1 - (f1**2 + f2**2) / den + torch.sin(2 * torch.pi * ((f1**2 - f2**2) / den + 1))
        return self._pack_population(x, torch.column_stack([f1, f2, f3]), c.unsqueeze(1))


CF_CLASSES = {cls.name: cls for cls in [CF1, CF2, CF3, CF4, CF5, CF6, CF7, CF8, CF9, CF10]}
