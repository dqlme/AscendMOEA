"""Implementation helpers for the nsga2 algorithm."""

from __future__ import annotations

import torch

from ..core import Algorithm, Population, Problem
from ..utils import crowding_distance, merge_pop, nd_sort, operator_ga, tournament_selection


class NSGAII(Algorithm):
    """Select solutions by constrained nondominated rank and crowding distance."""

    name = "NSGAII"

    def _environmental(self, pop: Population, n: int):
        # Constraints affect nondominated ranks, while crowding is measured
        # in the original objective space to preserve the reference semantics.
        """Select survivors by constrained front rank and crowding distance."""
        front_no, max_f = nd_sort(pop.objs, pop.cons, n_sort=n)
        crowd_dis = crowding_distance(pop.objs, front_no)

        selected = front_no < max_f
        remaining = n - int(selected.sum().item())
        if remaining > 0:
            last = torch.where(front_no == max_f)[0]
            rank = torch.argsort(crowd_dis[last], descending=True, stable=True)
            selected[last[rank[:remaining]]] = True

        survivors = torch.where(selected)[0]
        return (
            Population(pop.decs[survivors], pop.objs[survivors], pop.cons[survivors]),
            front_no[survivors],
            crowd_dis[survivors],
        )

    def _solve(self, problem: Problem, output, should_stop, should_pause) -> None:
        """Execute the vectorized evolutionary loop."""
        torch.set_default_dtype(problem.dtype)

        pop = problem.initialization()
        pop, front_no, crowd_dis = self._environmental(pop, problem.N)

        while self.not_terminated(problem, pop, output, should_stop, should_pause):
            mating = tournament_selection(2, problem.N, front_no, -crowd_dis)
            offspring = problem.evaluation(operator_ga(problem, pop.decs[mating]))
            pop, front_no, crowd_dis = self._environmental(merge_pop(pop, offspring), problem.N)
