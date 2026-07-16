"""Call public tensorized operators without running a complete algorithm."""

import torch

from ascendmoea.operators import (
    crowding_distance,
    nd_sort,
    operator_ga,
    tournament_selection,
)
from ascendmoea.problems import DTLZ2


def main() -> None:
    """Select parents and generate one vectorized offspring batch."""
    problem = DTLZ2(n=100, m=3, max_fe=10_000)
    population = problem.initialization()
    front_no, _ = nd_sort(population.objs, population.cons, n_sort=problem.N)
    crowding = crowding_distance(population.objs, front_no)
    parents = tournament_selection(2, problem.N, front_no, -crowding)
    offspring_dec = operator_ga(problem, population.decs[parents])
    offspring = problem.evaluate(offspring_dec)
    assert torch.all(torch.isfinite(offspring.objs))
    print(offspring.objs.shape)


if __name__ == "__main__":
    main()
