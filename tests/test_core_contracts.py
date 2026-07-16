"""Contract tests for core population and problem abstractions."""

import pytest
import torch

from ascendmoea import Population, Problem


class InvalidObjectiveProblem(Problem):
    """Problem that intentionally violates the objective shape contract."""

    def __init__(self) -> None:
        """Create a two-objective contract fixture."""
        super().__init__(n=4, m=2, d=3, max_fe=4)

    def setting(self) -> None:
        """Set unit-box decision bounds."""
        self.lower = torch.zeros(self.D, dtype=self.dtype, device=self.device)
        self.upper = torch.ones(self.D, dtype=self.dtype, device=self.device)

    def cal_obj(self, x: torch.Tensor) -> torch.Tensor:
        """Return one column instead of the configured two columns."""
        return x[:, :1]


def test_base_problem_is_abstract() -> None:
    """Prevent a bare problem from silently evaluating zero objectives."""
    with pytest.raises(TypeError):
        Problem()


def test_population_rejects_inconsistent_batch_shapes() -> None:
    """Reject fields that do not describe the same individual batch."""
    with pytest.raises(ValueError, match="same number of rows"):
        Population(torch.zeros((2, 3)), torch.zeros((3, 2)), torch.zeros((2, 1)))


def test_problem_rejects_invalid_decision_and_objective_shapes() -> None:
    """Report extension contract violations before incrementing evaluations."""
    problem = InvalidObjectiveProblem()
    with pytest.raises(ValueError, match="decisions must have shape"):
        problem.evaluate(torch.zeros((4, 2)))
    with pytest.raises(ValueError, match="cal_obj must return shape"):
        problem.evaluate(torch.zeros((4, 3)))
    assert problem.FE == 0
