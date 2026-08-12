"""Regression tests for degenerate population geometry."""

import os

import torch

from ascendmoea import Population
from ascendmoea.algorithms.bico import _cosine_similarity as bico_cosine
from ascendmoea.algorithms.bico import _maximum_similarity_candidates
from ascendmoea.algorithms.cmoeams import _cal_sde, _cosine_distance
from ascendmoea.algorithms.mmo_common import _harmonic_crowding, _local_convergence
from ascendmoea.algorithms.ts_sparse_ea import TSSparseEA
from ascendmoea.compiled_truncation import is_available, sequential_recurrence
from ascendmoea.utils import (
    pairwise_dist,
    sequential_distance_truncation,
    sequential_euclidean_truncation,
)


def test_cosine_helpers_accept_zero_vectors() -> None:
    """Keep similarities finite when masks or normalized points are all zero."""
    points = torch.tensor([[0.0, 0.0], [1.0, 0.0]])
    mask = torch.tensor([[False, False], [True, False]])
    assert torch.isfinite(bico_cosine(points)).all()
    assert torch.isfinite(_cosine_distance(points, points)).all()
    assert torch.isfinite(TSSparseEA._cosine_similarity(mask, points)).all()


def test_bico_maximum_similarity_ignores_nan_rows() -> None:
    """Match MATLAB max semantics for undefined zero-vector similarities."""
    cosine = torch.tensor(
        [
            [torch.nan, torch.nan, torch.nan],
            [torch.nan, 0.0, 0.75],
            [torch.nan, 0.75, 0.0],
        ]
    )
    expected = torch.tensor([[1, 2], [2, 1]])
    assert torch.equal(_maximum_similarity_candidates(cosine), expected)


def test_density_helpers_accept_constant_populations() -> None:
    """Keep diversity and convergence scores finite for zero-span batches."""
    decisions = torch.ones((4, 3))
    objectives = torch.ones((4, 2))
    constraints = torch.zeros((4, 1))
    population = Population(decisions, objectives, constraints)
    assert torch.isfinite(_cal_sde(objectives)).all()
    assert torch.isfinite(_harmonic_crowding(decisions)).all()
    assert torch.isfinite(_local_convergence(population)).all()


def test_sequential_truncation_places_nan_after_infinity() -> None:
    """Match ascending stable row ordering for all-NaN pair distances."""
    distance = torch.full((4, 4), torch.nan, dtype=torch.float64)
    distance.fill_diagonal_(0.0)
    previous = os.environ.get("ASCENDMOEA_COMPILED_TRUNCATION")
    os.environ["ASCENDMOEA_COMPILED_TRUNCATION"] = "off"
    try:
        reference = sequential_distance_truncation(distance, 2)
    finally:
        if previous is None:
            os.environ.pop("ASCENDMOEA_COMPILED_TRUNCATION", None)
        else:
            os.environ["ASCENDMOEA_COMPILED_TRUNCATION"] = previous
    assert torch.equal(torch.where(reference)[0], torch.tensor([0, 1]))

    if is_available():
        ordered = distance.clone()
        ordered.fill_diagonal_(torch.inf)
        values, indices = torch.sort(ordered, dim=1, stable=True)
        compiled = sequential_recurrence(values, indices, 2)
        assert compiled is not None
        assert torch.equal(torch.where(compiled)[0], torch.tensor([0, 1]))


def test_multi_space_truncation_matches_explicit_distance_sum() -> None:
    """Preserve exact deletion semantics when combining distance spaces."""
    first = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 2.0], [3.0, 3.0]])
    second = torch.tensor([[0.0], [2.0], [1.0], [4.0]])
    expected_distance = pairwise_dist(first, first) + pairwise_dist(second, second)
    expected = sequential_distance_truncation(expected_distance, 2)
    actual = sequential_euclidean_truncation(first, second, delete_count=2)
    assert torch.equal(actual, expected)
