"""Repeated-run experiment helpers and statistical summaries."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import torch
from scipy.stats import friedmanchisquare, ranksums, wilcoxon

from .core import Algorithm, Problem
from .metrics import METRICS
from .workflow import RunConfig, Workflow


@dataclass
class RunRecord:
    """Store metrics for one algorithm/problem repetition."""

    algorithm: str
    problem: str
    run: int
    metrics: dict[str, float]


def run_single(
    alg: Algorithm,
    pro: Problem,
    output_callback=None,
    should_stop_cb=None,
    *,
    device=None,
    seed: int | None = None,
    cpu_threads: int | None = None,
):
    """Run one synchronized experiment and calculate all available metrics."""
    result = Workflow(
        alg,
        pro,
        config=RunConfig(device=device, seed=seed, cpu_threads=cpu_threads),
    ).run(output=output_callback, should_stop=should_stop_cb)
    runtime = result.elapsed_seconds
    final = alg.result[-1][1]
    metric_values = {"runtime": runtime}
    for k, (fn, _) in METRICS.items():
        if k == "runtime" or fn is None:
            continue
        try:
            metric_values[k] = float(fn(final, pro.optimum))
        except Exception:
            metric_values[k] = float(torch.nan)
    return metric_values, alg


def run_experiment(alg_factory, pro_factory, n_runs: int):
    """Execute independent repetitions from algorithm and problem factories."""
    records: list[RunRecord] = []
    for r in range(1, n_runs + 1):
        alg = alg_factory()
        pro = pro_factory()
        values, _ = run_single(alg, pro)
        records.append(RunRecord(type(alg).__name__, type(pro).__name__, r, values))
    return records


def records_to_frame(records: list[RunRecord]) -> pd.DataFrame:
    """Convert run records to a tidy pandas data frame."""
    rows = []
    for rec in records:
        row = {"Algorithm": rec.algorithm, "Problem": rec.problem, "Run": rec.run}
        row.update(rec.metrics)
        rows.append(row)
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame, metric: str, style: str = "mean_std") -> pd.DataFrame:
    """Aggregate a metric by problem and algorithm."""
    g = df.groupby(["Problem", "Algorithm"])[metric]
    if style == "mean":
        s = g.mean().to_frame("value")
    elif style == "median_iqr":
        s = pd.concat(
            [g.median().rename("median"), (g.quantile(0.75) - g.quantile(0.25)).rename("iqr")],
            axis=1,
        )
    else:
        s = pd.concat([g.mean().rename("mean"), g.std().rename("std")], axis=1)
    return s.reset_index()


def significance_tests(df: pd.DataFrame, metric: str, test: str, baseline_alg: str | None = None):
    """Run Friedman, signed-rank, or rank-sum tests on repeated results."""
    out = []
    pivot = df.pivot_table(
        index=["Problem", "Run"], columns="Algorithm", values=metric, aggfunc="mean"
    )
    algs = [a for a in pivot.columns if baseline_alg is None or a != baseline_alg]
    if test == "friedman":
        vals = [pivot[a].dropna().values for a in pivot.columns]
        minlen = min(len(v) for v in vals)
        if minlen < 2:
            return []
        stat, p = friedmanchisquare(*[v[:minlen] for v in vals])
        return [{"test": "friedman", "stat": float(stat), "p": float(p)}]

    if baseline_alg is None or baseline_alg not in pivot.columns:
        return []
    base = pivot[baseline_alg]
    for a in algs:
        m = pd.concat([base, pivot[a]], axis=1).dropna()
        if len(m) < 2:
            continue
        if test == "signrank":
            stat, p = wilcoxon(m.iloc[:, 0], m.iloc[:, 1])
        else:
            stat, p = ranksums(m.iloc[:, 0], m.iloc[:, 1])
        out.append({"algorithm": a, "stat": float(stat), "p": float(p)})
    return out
