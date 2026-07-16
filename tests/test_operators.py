"""Behavioral tests for public tensorized population operators."""

import pytest
import torch

from ascendmoea.operators import (
    crowding_distance,
    domination_count,
    nd_sort,
    non_dominated_mask,
    operator_ga_encoded,
    pairwise_dist,
)
from ascendmoea.problems import DTLZ2


def test_dominance_and_front_assignment() -> None:
    """Identify two nondominated points and one dominated point."""
    objectives = torch.tensor([[1.0, 2.0], [2.0, 1.0], [3.0, 3.0]])
    expected = torch.tensor([True, True, False])
    assert torch.equal(non_dominated_mask(objectives), expected)
    assert torch.equal(domination_count(objectives), torch.tensor([0, 0, 2]))
    fronts, last = nd_sort(objectives)
    assert torch.equal(fronts, torch.tensor([1.0, 1.0, 2.0]))
    assert last == 2
    crowding = crowding_distance(objectives, fronts)
    assert torch.isinf(crowding[:2]).all()


def test_pairwise_distance_is_symmetric() -> None:
    """Preserve exact zeros and symmetry for a self-distance matrix."""
    points = torch.tensor([[0.0, 0.0], [3.0, 4.0]])
    distance = pairwise_dist(points, points)
    assert torch.equal(distance.diag(), torch.zeros(2))
    expected = torch.tensor(
        [[0.0, 5.0], [5.0, 0.0]],
        dtype=distance.dtype,
        device=distance.device,
    )
    assert torch.allclose(distance, expected)


def test_encoded_genetic_operator_accepts_real_problems() -> None:
    """Exercise the real-variable proxy through the public encoded operator."""
    problem = DTLZ2(n=10, m=3, d=8, max_fe=20)
    parents = problem.initialization().decs
    offspring = operator_ga_encoded(problem, parents)
    assert offspring.shape == parents.shape
    assert offspring.dtype == parents.dtype
    assert offspring.device == parents.device
    assert torch.all(offspring >= problem.lower)
    assert torch.all(offspring <= problem.upper)


def test_public_operators_reject_invalid_shapes() -> None:
    """Return descriptive errors instead of relying on broadcast failures."""
    problem = DTLZ2(n=4, m=3, d=6, max_fe=8)
    with pytest.raises(ValueError, match="pop_obj"):
        nd_sort(torch.zeros(4))
    with pytest.raises(ValueError, match="parent_dec"):
        operator_ga_encoded(problem, torch.zeros((4, 5)))
