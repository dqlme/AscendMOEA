"""Tests for CPU and accelerator release-report comparison."""

from copy import deepcopy

import pytest

from tools.compare_validation_reports import compare_reports


def _report() -> dict[str, object]:
    """Create a minimal successful release report."""
    return {
        "schema_version": 1,
        "package": "AscendMOEA",
        "version": "0.1.0",
        "device": "cpu",
        "counts": {"algorithms": 1, "metrics": 1},
        "status": "ok",
        "results": [
            {
                "category": "algorithm",
                "name": "Example",
                "status": "ok",
                "details": {
                    "problem": "ExampleProblem",
                    "max_fe": 100,
                    "FE": 104,
                    "budget_reached": True,
                    "population": 20,
                    "algorithm_options": {},
                },
            },
            {
                "category": "metrics",
                "name": "public_metric_surface",
                "status": "ok",
                "details": {
                    "metric_count": 2,
                    "input": "deterministic_linspace",
                    "values": {"HV": 0.5, "IGD": 0.25},
                },
            },
        ],
    }


def test_comparison_accepts_stochastic_evaluation_count_difference() -> None:
    """Do not require independent device RNGs to follow the same trajectory."""
    cpu = _report()
    accelerator = deepcopy(cpu)
    accelerator["device"] = "npu:0"
    accelerator["results"][0]["details"]["FE"] = 107

    summary = compare_reports(
        cpu,
        accelerator,
        relative_tolerance=1e-5,
        absolute_tolerance=1e-6,
    )

    assert summary["status"] == "ok"
    assert summary["components"] == 2


def test_comparison_rejects_budget_or_metric_mismatch() -> None:
    """Keep configured budgets and deterministic metric values strict."""
    cpu = _report()
    accelerator = deepcopy(cpu)
    accelerator["results"][0]["details"]["max_fe"] = 101
    with pytest.raises(ValueError, match="max_fe"):
        compare_reports(
            cpu,
            accelerator,
            relative_tolerance=1e-5,
            absolute_tolerance=1e-6,
        )

    accelerator = deepcopy(cpu)
    accelerator["results"][1]["details"]["values"]["HV"] = 0.6
    with pytest.raises(ValueError, match="metric HV"):
        compare_reports(
            cpu,
            accelerator,
            relative_tolerance=1e-5,
            absolute_tolerance=1e-6,
        )
