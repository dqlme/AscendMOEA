"""Repeated-run experiment helpers and statistical summaries."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import pandas as pd
import torch
from scipy.stats import friedmanchisquare, ranksums, wilcoxon

from .core import Algorithm, Problem
from .metrics import METRICS
from .workflow import RunConfig, Workflow


@dataclass(frozen=True)
class RunRecord:
    """Store metrics and identifiers for one independent repetition."""

    algorithm: str
    problem: str
    run: int
    metrics: dict[str, float]


def run_single(
    alg: Algorithm,
    pro: Problem,
    output_callback: Callable[[Algorithm, Problem], None] | None = None,
    should_stop_cb: Callable[[], bool] | None = None,
    *,
    device: torch.device | str | None = None,
    seed: int | None = None,
    cpu_threads: int | None = None,
) -> tuple[dict[str, float], Algorithm]:
    """Run one synchronized experiment and calculate compatible metrics.

    Objective-space indicators use ``problem.optimum``. Decision-space
    indicators use ``problem.ps`` and return ``NaN`` when the problem does not
    expose a reference Pareto set.
    """
    result = Workflow(
        alg,
        pro,
        config=RunConfig(device=device, seed=seed, cpu_threads=cpu_threads),
    ).run(output=output_callback, should_stop=should_stop_cb)
    final = result.population
    metric_values = {"runtime": result.elapsed_seconds}
    decision_space_metrics = {"IGDX", "CR", "PSP"}
    for name, (function, _) in METRICS.items():
        if name == "runtime" or function is None:
            continue
        reference = getattr(pro, "ps", None) if name in decision_space_metrics else pro.optimum
        if reference is None:
            metric_values[name] = float("nan")
            continue
        try:
            metric_values[name] = float(function(final, reference))
        except (IndexError, RuntimeError, TypeError, ValueError):
            metric_values[name] = float("nan")
    return metric_values, alg


def run_experiment(
    alg_factory: Callable[[], Algorithm],
    pro_factory: Callable[[], Problem],
    n_runs: int,
    *,
    device: torch.device | str | None = None,
    base_seed: int | None = None,
    cpu_threads: int | None = None,
) -> list[RunRecord]:
    """Execute reproducible repetitions from algorithm and problem factories.

    ``base_seed`` assigns consecutive seeds to runs. Pass ``None`` only when
    intentionally requesting uncontrolled random streams.
    """
    if n_runs < 1:
        raise ValueError("n_runs must be at least one")
    records: list[RunRecord] = []
    for r in range(1, n_runs + 1):
        alg = alg_factory()
        pro = pro_factory()
        seed = None if base_seed is None else int(base_seed) + r - 1
        values, _ = run_single(
            alg,
            pro,
            device=device,
            seed=seed,
            cpu_threads=cpu_threads,
        )
        records.append(
            RunRecord(
                str(getattr(alg, "name", type(alg).__name__)),
                str(getattr(pro, "name", type(pro).__name__)),
                r,
                values,
            )
        )
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
    """Aggregate a metric by problem and algorithm.

    Args:
        df: Tidy experiment frame produced by :func:`records_to_frame`.
        metric: Metric column to summarize.
        style: One of ``"mean"``, ``"mean_std"``, or ``"median_iqr"``.
    """
    if style not in {"mean", "mean_std", "median_iqr"}:
        raise ValueError("style must be mean, mean_std, or median_iqr")
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


def significance_tests(
    df: pd.DataFrame,
    metric: str,
    test: str,
    baseline_alg: str | None = None,
) -> list[dict[str, Any]]:
    """Run Friedman, paired Wilcoxon, or rank-sum tests.

    Friedman and Wilcoxon calculations use only aligned problem/run rows.
    ``baseline_alg`` is required for pairwise tests.
    """
    aliases = {"signrank": "wilcoxon", "ranksums": "ranksum"}
    test = aliases.get(test, test)
    if test not in {"friedman", "wilcoxon", "ranksum"}:
        raise ValueError("test must be friedman, wilcoxon, or ranksum")
    out: list[dict[str, Any]] = []
    pivot = df.pivot_table(
        index=["Problem", "Run"], columns="Algorithm", values=metric, aggfunc="mean"
    )
    if test == "friedman":
        aligned = pivot.dropna(axis=0, how="any")
        if aligned.shape[1] < 3 or aligned.shape[0] < 2:
            return []
        stat, p = friedmanchisquare(*[aligned[column].to_numpy() for column in aligned])
        return [{"test": "friedman", "stat": float(stat), "p": float(p)}]

    if baseline_alg is None or baseline_alg not in pivot.columns:
        return []
    algs = [algorithm for algorithm in pivot.columns if algorithm != baseline_alg]
    base = pivot[baseline_alg]
    for algorithm in algs:
        aligned = pd.concat([base, pivot[algorithm]], axis=1).dropna()
        if len(aligned) < 2:
            continue
        if test == "wilcoxon":
            difference = aligned.iloc[:, 0] - aligned.iloc[:, 1]
            if bool((difference == 0).all()):
                stat, p = 0.0, 1.0
            else:
                stat, p = wilcoxon(aligned.iloc[:, 0], aligned.iloc[:, 1])
        else:
            stat, p = ranksums(aligned.iloc[:, 0], aligned.iloc[:, 1])
        out.append({"algorithm": algorithm, "stat": float(stat), "p": float(p)})
    return out


__all__ = [
    "RunRecord",
    "records_to_frame",
    "run_experiment",
    "run_single",
    "significance_tests",
    "summarize",
]
