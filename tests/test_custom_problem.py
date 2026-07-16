"""Tests for the vectorized custom-problem contract."""

import torch

from ascendmoea import Problem


class TwoSphere(Problem):
    """Small deterministic problem used by the contract test."""

    def __init__(self) -> None:
        """Initialize a four-dimensional test problem."""
        super().__init__(n=8, m=2, d=4, max_fe=16)

    def setting(self) -> None:
        """Set symmetric continuous bounds."""
        self.lower = torch.full((self.D,), -1.0, dtype=self.dtype, device=self.device)
        self.upper = torch.full((self.D,), 1.0, dtype=self.dtype, device=self.device)

    def cal_obj(self, x: torch.Tensor) -> torch.Tensor:
        """Evaluate two sphere objectives in one tensor operation."""
        return torch.column_stack((x.square().sum(1), (x - 1).square().sum(1)))


class RepairedProblem(TwoSphere):
    """Problem that projects the first two variables onto a unit-sum plane."""

    def cal_dec(self, x: torch.Tensor) -> torch.Tensor:
        """Repair the first two variables without changing the batch shape."""
        repaired = x.clone()
        repaired[:, :2] = repaired[:, :2] / repaired[:, :2].sum(dim=1, keepdim=True)
        return repaired


def test_custom_problem_evaluates_a_batch() -> None:
    """Ensure batch shape, constraints, and evaluation counting are correct."""
    problem = TwoSphere()
    decisions = torch.zeros((8, 4), dtype=problem.dtype)
    population = problem.evaluate(decisions)
    assert population.objs.shape == (8, 2)
    assert population.cons.shape == (8, 1)
    assert problem.FE == 8
    assert torch.all(population.feasible)


def test_custom_problem_can_repair_a_complete_batch() -> None:
    """Run the public decision-repair hook after generic bound handling."""
    problem = RepairedProblem()
    decisions = torch.full((3, 4), 0.25, dtype=problem.dtype)
    population = problem.evaluate(decisions)
    assert population.decs.shape == decisions.shape
    assert torch.allclose(
        population.decs[:, :2].sum(dim=1),
        torch.ones(3, dtype=population.decs.dtype, device=population.decs.device),
    )
