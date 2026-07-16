"""Implementation helpers for the hrea algorithm."""

from __future__ import annotations

import torch

from ..core import Algorithm, Problem
from ..utils import merge_pop, operator_ga, tournament_selection
from .mmo_common import _hrea_archive_update, _hrea_environmental


class HREA(Algorithm):
    """Implement the HREA multi-objective evolutionary algorithm."""

    name = "HREA"

    def __init__(self, eps: float = 0.3, p: float = 0.5, **kwargs):
        """Initialize the instance."""
        super().__init__(**kwargs)
        self.eps = float(eps)
        self.p = float(p)

    def _solve(self, problem: Problem, output, should_stop, should_pause) -> None:
        """Execute the vectorized evolutionary loop."""
        torch.set_default_dtype(problem.dtype)
        pop = problem.initialization()
        _, crowd1 = _hrea_environmental(pop, problem.N)
        archive, crowd2 = _hrea_archive_update(pop, problem.N, self.eps, 0.0)
        while self.not_terminated(problem, archive, output, should_stop, should_pause):
            if (
                problem.FE >= problem.max_fe * 0.5
                and float(torch.rand((), device=problem.device).item()) < self.p
            ):
                mating = tournament_selection(2, problem.N, -crowd2)
                parent = archive.decs[mating]
            else:
                mating = tournament_selection(2, problem.N, -crowd1)
                parent = pop.decs[mating]
            off_dec = operator_ga(problem, parent)
            off = problem.evaluation(off_dec)
            pop, crowd1 = _hrea_environmental(merge_pop(pop, off), problem.N)
            archive, crowd2 = _hrea_archive_update(
                merge_pop(archive, off), problem.N, self.eps, problem.FE / problem.max_fe
            )
