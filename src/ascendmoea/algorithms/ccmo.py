"""Implementation helpers for the ccmo algorithm."""

from __future__ import annotations

import torch

from ..core import Algorithm, Problem
from ..utils import merge_pop, operator_de, operator_ga, tournament_selection
from .spea_common import cal_fitness, environmental_selection


class CCMO(Algorithm):
    """Co-evolve constrained and unconstrained auxiliary populations."""

    name = "CCMO"

    def __init__(self, op_type: int = 1, **kwargs):
        """Configure the variation operator.

        Args:
            op_type: ``1`` selects GA and ``2`` selects DE.
            **kwargs: Arguments forwarded to :class:`~ascendmoea.Algorithm`.
        """
        super().__init__(**kwargs)
        self.op_type = int(op_type)
        if self.op_type not in {1, 2}:
            raise ValueError("op_type must be 1 (GA) or 2 (DE)")

    def _solve(self, problem: Problem, output, should_stop, should_pause) -> None:
        """Execute the vectorized evolutionary loop."""
        torch.set_default_dtype(problem.dtype)
        population1 = problem.initialization()
        population2 = problem.initialization()
        fitness1 = cal_fitness(population1.objs, population1.cons)
        fitness2 = cal_fitness(population2.objs)

        while self.not_terminated(problem, population1, output, should_stop, should_pause):
            if self.op_type == 1:
                mating1 = tournament_selection(2, problem.N, fitness1)
                mating2 = tournament_selection(2, problem.N, fitness2)
                offspring1 = problem.evaluation(
                    operator_ga(problem, population1.decs[mating1], half=True)
                )
                offspring2 = problem.evaluation(
                    operator_ga(problem, population2.decs[mating2], half=True)
                )
            elif self.op_type == 2:
                mating1 = tournament_selection(2, 2 * problem.N, fitness1)
                mating2 = tournament_selection(2, 2 * problem.N, fitness2)
                offspring1 = problem.evaluation(
                    operator_de(
                        problem,
                        population1.decs,
                        population1.decs[mating1[: problem.N]],
                        population1.decs[mating1[problem.N :]],
                    )
                )
                offspring2 = problem.evaluation(
                    operator_de(
                        problem,
                        population2.decs,
                        population2.decs[mating2[: problem.N]],
                        population2.decs[mating2[problem.N :]],
                    )
                )
            else:
                raise ValueError(f"Unsupported CCMO operator type: {self.op_type}")

            merged = merge_pop(population1, offspring1, offspring2)
            population1, fitness1 = environmental_selection(
                merged,
                problem.N,
                use_constraints=True,
                sort_output=True,
            )
            merged = merge_pop(population2, offspring1, offspring2)
            population2, fitness2 = environmental_selection(
                merged,
                problem.N,
                use_constraints=False,
                sort_output=True,
            )
