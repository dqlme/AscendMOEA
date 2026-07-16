"""Tests for optional repeated-experiment and statistics helpers."""

import math

import pytest

from ascendmoea.algorithms import NSGAII
from ascendmoea.experiment import run_experiment, run_single, significance_tests, summarize
from ascendmoea.problems import MMF1

pandas = pytest.importorskip("pandas")


def test_run_single_uses_decision_space_reference_data() -> None:
    """Calculate IGDX-family metrics from a problem's Pareto set."""
    values, _ = run_single(
        NSGAII(save=1),
        MMF1(n=20, max_fe=40),
        device="cpu",
        seed=23,
        cpu_threads=1,
    )
    assert not math.isnan(values["IGDX"])
    assert not math.isnan(values["CR"])
    assert not math.isnan(values["PSP"])


def test_repeated_experiment_assigns_stable_run_ids() -> None:
    """Create independent instances and consecutive deterministic seeds."""
    records = run_experiment(
        lambda: NSGAII(save=1),
        lambda: MMF1(n=10, max_fe=20),
        2,
        device="cpu",
        base_seed=100,
        cpu_threads=1,
    )
    assert [record.run for record in records] == [1, 2]
    assert all(record.algorithm == "NSGAII" for record in records)


def test_statistics_validate_modes_and_align_friedman_rows() -> None:
    """Reject misspelled modes and ignore incomplete paired rows."""
    frame = pandas.DataFrame(
        [
            {"Problem": "P", "Run": 1, "Algorithm": "A", "IGD": 1.0},
            {"Problem": "P", "Run": 1, "Algorithm": "B", "IGD": 2.0},
            {"Problem": "P", "Run": 1, "Algorithm": "C", "IGD": 3.0},
            {"Problem": "P", "Run": 2, "Algorithm": "A", "IGD": 1.5},
            {"Problem": "P", "Run": 2, "Algorithm": "B", "IGD": 2.5},
            {"Problem": "P", "Run": 2, "Algorithm": "C", "IGD": 3.5},
            {"Problem": "P", "Run": 3, "Algorithm": "A", "IGD": 0.5},
        ]
    )
    assert significance_tests(frame, "IGD", "friedman")
    with pytest.raises(ValueError, match="style"):
        summarize(frame, "IGD", style="means")
    with pytest.raises(ValueError, match="test"):
        significance_tests(frame, "IGD", "unknown")
