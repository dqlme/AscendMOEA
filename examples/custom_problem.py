"""Define, register, and optimize a vectorized user problem."""

import torch

from ascendmoea import Problem, create_problem, optimize, register_problem
from ascendmoea.algorithms import NSGAII


@register_problem(name="BiSphere")
class BiSphere(Problem):
    """Two-objective sphere problem with separated optima."""

    name = "BiSphere"

    def __init__(self, n: int = 100, d: int = 10, max_fe: int = 10_000) -> None:
        """Initialize population size, dimension, and evaluation budget."""
        super().__init__(n=n, m=2, d=d, max_fe=max_fe)

    def setting(self) -> None:
        """Set continuous decision bounds."""
        self.lower = torch.full((self.D,), -5.0, dtype=self.dtype, device=self.device)
        self.upper = torch.full((self.D,), 5.0, dtype=self.dtype, device=self.device)

    def cal_obj(self, x: torch.Tensor) -> torch.Tensor:
        """Evaluate both objectives for the complete decision batch."""
        return torch.column_stack(
            [
                torch.sum(x.square(), dim=1),
                torch.sum((x - 2.0).square(), dim=1),
            ]
        )


def main() -> None:
    """Construct the custom problem from the registry and optimize it."""
    problem = create_problem("BiSphere", n=100, d=20, max_fe=10_000)
    result = optimize(NSGAII(), problem, device="cpu", seed=7)
    print(result.population.objs)


if __name__ == "__main__":
    main()
