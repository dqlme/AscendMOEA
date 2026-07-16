"""Regression tests for objective- and decision-space quality indicators."""

import math

import pytest
import torch

from ascendmoea import Population
from ascendmoea.metrics import coverage_rate, feasible_rate, gd, igd, igdx, psp


def _population() -> Population:
    """Create a small feasible approximation with known decision ranges."""
    return Population(
        decs=torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.float64),
        objs=torch.tensor([[0.0, 1.0], [1.0, 0.0]], dtype=torch.float64),
        cons=torch.zeros((2, 1), dtype=torch.float64),
    )


def test_exact_reference_sets_have_zero_distance_and_full_coverage() -> None:
    """Return exact values when objective and decision references match the population."""
    population = _population()
    assert igd(population, population.objs) == pytest.approx(0.0)
    assert gd(population, population.objs) == pytest.approx(0.0)
    assert igdx(population, population.decs) == pytest.approx(0.0)
    assert coverage_rate(population, population.decs) == pytest.approx(1.0)
    assert psp(population, population.decs) == pytest.approx(0.0)
    assert feasible_rate(population) == pytest.approx(1.0)


def test_missing_references_and_empty_populations_are_reported() -> None:
    """Use NaN for unavailable information instead of fabricating indicator values."""
    population = _population()
    assert math.isnan(igd(population, None))
    assert math.isnan(igdx(population, torch.empty((0, 2))))
    empty = Population(
        decs=torch.empty((0, 2)),
        objs=torch.empty((0, 2)),
        cons=torch.empty((0, 1)),
    )
    assert math.isnan(feasible_rate(empty))


def test_zero_decision_coverage_produces_infinite_psp() -> None:
    """Preserve the PSP definition when approximation and reference ranges do not overlap."""
    population = _population()
    distant_reference = torch.tensor([[2.0, 2.0], [3.0, 3.0]], dtype=torch.float64)
    assert coverage_rate(population, distant_reference) == 0.0
    assert math.isinf(psp(population, distant_reference))
