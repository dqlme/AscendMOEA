"""Build a custom algorithm from public population operators."""

import torch

from ascendmoea import Algorithm, Population, Problem, optimize, register_algorithm
from ascendmoea.operators import crowding_distance, merge_pop, nd_sort
from ascendmoea.problems import DTLZ2


def environmental_selection(population: Population, size: int) -> Population:
    """Keep the best fronts and truncate the final front by crowding distance."""
    front_no, last_front = nd_sort(
        population.objs,
        population.cons,
        n_sort=size,
    )
    crowding = crowding_distance(population.objs, front_no)
    selected = front_no < last_front
    remaining = size - int(selected.sum().item())
    if remaining > 0:
        candidates = torch.where(front_no == last_front)[0]
        order = torch.argsort(crowding[candidates], descending=True, stable=True)
        selected[candidates[order[:remaining]]] = True
    return population[selected]


@register_algorithm(name="VectorizedRandomSearch")
class VectorizedRandomSearch(Algorithm):
    """Generate and select one full random batch per generation."""

    name = "VectorizedRandomSearch"

    def _solve(self, problem: Problem, output, should_stop, should_pause) -> None:
        """Execute the batch random-search loop."""
        population = problem.initialization()
        while self.not_terminated(
            problem,
            population,
            output,
            should_stop,
            should_pause,
        ):
            offspring = problem.initialization(problem.N)
            population = environmental_selection(
                merge_pop(population, offspring),
                problem.N,
            )


def main() -> None:
    """Run the custom algorithm on a built-in problem."""
    result = optimize(
        VectorizedRandomSearch(),
        DTLZ2(n=100, m=3, max_fe=10_000),
        device="cpu",
        seed=11,
    )
    print(result.population.objs)


if __name__ == "__main__":
    main()
