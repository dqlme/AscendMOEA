"""Regression tests for degenerate population geometry."""

import torch

from ascendmoea import Population
from ascendmoea.algorithms.bico import _cosine_similarity as bico_cosine
from ascendmoea.algorithms.cmoeams import _cal_sde, _cosine_distance
from ascendmoea.algorithms.mmo_common import _harmonic_crowding, _local_convergence
from ascendmoea.algorithms.ts_sparse_ea import TSSparseEA


def test_cosine_helpers_accept_zero_vectors() -> None:
    """Keep similarities finite when masks or normalized points are all zero."""
    points = torch.tensor([[0.0, 0.0], [1.0, 0.0]])
    mask = torch.tensor([[False, False], [True, False]])
    assert torch.isfinite(bico_cosine(points)).all()
    assert torch.isfinite(_cosine_distance(points, points)).all()
    assert torch.isfinite(TSSparseEA._cosine_similarity(mask, points)).all()


def test_density_helpers_accept_constant_populations() -> None:
    """Keep diversity and convergence scores finite for zero-span batches."""
    decisions = torch.ones((4, 3))
    objectives = torch.ones((4, 2))
    constraints = torch.zeros((4, 1))
    population = Population(decisions, objectives, constraints)
    assert torch.isfinite(_cal_sde(objectives)).all()
    assert torch.isfinite(_harmonic_crowding(decisions)).all()
    assert torch.isfinite(_local_convergence(population)).all()
