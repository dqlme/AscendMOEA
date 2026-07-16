"""Implementation helpers for the moea_psl algorithm."""

from __future__ import annotations

import math

import torch

from ..core import Algorithm, Population, Problem
from ..utils import crowding_distance, merge_pop, nd_sort, tournament_selection
from .sparse_common import subset_population, unique_row_indices


@torch.jit.script
def _train_rbm_batch1(
    x: torch.Tensor,
    weight: torch.Tensor,
    visible_bias: torch.Tensor,
    hidden_bias: torch.Tensor,
    epochs: int,
    momentum: float,
    learn_rate: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute train rbm batch1."""
    weight_increment = torch.zeros_like(weight)
    visible_increment = torch.zeros_like(visible_bias)
    hidden_increment = torch.zeros_like(hidden_bias)
    for _ in range(epochs):
        order = torch.randperm(x.shape[0], device=x.device)
        shuffled = x.index_select(0, order)
        for row in range(shuffled.shape[0]):
            batch = shuffled[row]
            positive_prob = torch.sigmoid(torch.mv(weight.T, batch) + hidden_bias)
            positive_state = positive_prob > torch.rand_like(positive_prob)
            negative_data_prob = torch.sigmoid(
                torch.mv(weight, positive_state.to(x.dtype)) + visible_bias
            )
            negative_data = negative_data_prob > torch.rand_like(negative_data_prob)
            negative_hidden_prob = torch.sigmoid(
                torch.mv(weight.T, negative_data.to(x.dtype)) + hidden_bias
            )
            weight_increment = momentum * weight_increment + learn_rate * (
                torch.outer(batch, positive_prob)
                - torch.outer(negative_data_prob, negative_hidden_prob)
            )
            # A batch of one uses scalar aggregate updates for both biases.
            visible_increment = momentum * visible_increment + learn_rate * (
                batch.sum() - negative_data.to(x.dtype).sum()
            )
            hidden_increment = momentum * hidden_increment + learn_rate * (
                positive_prob.sum() - negative_hidden_prob.sum()
            )
            weight = weight + weight_increment
            visible_bias = visible_bias + visible_increment
            hidden_bias = hidden_bias + hidden_increment
    return weight, visible_bias, hidden_bias


class _RBM:
    """Implement the _RBM multi-objective evolutionary algorithm."""

    def __init__(self, visible: int, hidden: int, problem: Problem):
        """Initialize the instance."""
        self.visible = int(visible)
        self.hidden = int(hidden)
        self.epoch = 10
        self.momentum = 0.5
        self.learn_rate = 0.1
        self.weight = 0.1 * torch.randn(
            (visible, hidden), dtype=problem.dtype, device=problem.device
        )
        self.visible_bias = torch.zeros(visible, dtype=problem.dtype, device=problem.device)
        self.hidden_bias = torch.zeros(hidden, dtype=problem.dtype, device=problem.device)

    def train(self, x: torch.Tensor) -> None:
        """Compute train."""
        x = x.to(self.weight.dtype)
        momentum = 0.9 if self.epoch > 5 else self.momentum
        self.weight, self.visible_bias, self.hidden_bias = _train_rbm_batch1(
            x,
            self.weight,
            self.visible_bias,
            self.hidden_bias,
            self.epoch,
            momentum,
            self.learn_rate,
        )

    def reduce(self, x: torch.Tensor) -> torch.Tensor:
        """Compute reduce."""
        probability = torch.sigmoid(x.to(self.weight.dtype) @ self.weight + self.hidden_bias)
        return probability > torch.rand_like(probability)

    def recover(self, hidden: torch.Tensor) -> torch.Tensor:
        """Compute recover."""
        probability = torch.sigmoid(
            hidden.to(self.weight.dtype) @ self.weight.T + self.visible_bias
        )
        return probability > torch.rand_like(probability)


class _DAE:
    """Implement the _DAE multi-objective evolutionary algorithm."""

    def __init__(self, visible: int, hidden: int, population_size: int, problem: Problem):
        """Initialize the instance."""
        self.visible = int(visible)
        self.hidden = int(hidden)
        self.epoch = 10
        self.batch_size = int(population_size)
        self.mask_fraction = 0.5
        self.momentum = 0.5
        self.learn_rate = 0.1
        scale = 8 * math.sqrt(6 / (hidden + visible))
        self.wa = (
            torch.rand((hidden, visible + 1), dtype=problem.dtype, device=problem.device) - 0.5
        ) * scale
        self.wb = (
            torch.rand((visible, hidden + 1), dtype=problem.dtype, device=problem.device) - 0.5
        ) * scale
        self.lower = torch.empty(0, dtype=problem.dtype, device=problem.device)
        self.upper = torch.empty(0, dtype=problem.dtype, device=problem.device)

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Compute normalize."""
        span = self.upper - self.lower
        return torch.where(span != 0, (x - self.lower) / span, torch.zeros_like(x))

    def train(self, x: torch.Tensor) -> None:
        """Compute train."""
        self.lower = x.min(dim=0).values
        self.upper = x.max(dim=0).values
        normalized = self._normalize(x)
        velocity_a = torch.zeros_like(self.wa)
        velocity_b = torch.zeros_like(self.wb)
        theta = torch.rand_like(normalized) > self.mask_fraction
        corrupted = normalized * theta
        corrupted = torch.cat(
            [torch.ones((x.shape[0], 1), dtype=x.dtype, device=x.device), corrupted],
            dim=1,
        )
        for _ in range(self.epoch):
            order = torch.randperm(x.shape[0], device=x.device)
            batch_x = corrupted[order]
            batch_y = normalized[order]
            hidden = torch.sigmoid(batch_x @ self.wa.T)
            hidden_bias = torch.cat(
                [torch.ones((x.shape[0], 1), dtype=x.dtype, device=x.device), hidden],
                dim=1,
            )
            reconstructed = torch.sigmoid(hidden_bias @ self.wb.T)
            error = batch_y - reconstructed
            delta_output = -error * reconstructed * (1 - reconstructed)
            activation_derivative = hidden_bias * (1 - hidden_bias)
            delta_hidden = delta_output @ self.wb * activation_derivative
            gradient_a = delta_hidden[:, 1:].T @ batch_x / x.shape[0]
            gradient_b = delta_output.T @ hidden_bias / x.shape[0]
            velocity_a = self.momentum * velocity_a + self.learn_rate * gradient_a
            velocity_b = self.momentum * velocity_b + self.learn_rate * gradient_b
            self.wa -= velocity_a
            self.wb -= velocity_b

    def reduce(self, x: torch.Tensor) -> torch.Tensor:
        """Compute reduce."""
        normalized = self._normalize(x)
        return torch.sigmoid(normalized @ self.wa[:, 1:].T + self.wa[:, 0])

    def recover(self, hidden: torch.Tensor) -> torch.Tensor:
        """Compute recover."""
        normalized = torch.sigmoid(hidden @ self.wb[:, 1:].T + self.wb[:, 0])
        return normalized * (self.upper - self.lower) + self.lower


class MOEAPSL(Algorithm):
    """Learn sparse Pareto-set structure with generative latent models."""

    name = "MOEA-PSL"

    @staticmethod
    def _latin(n: int, d: int, problem: Problem) -> torch.Tensor:
        """Compute latin."""
        order = torch.argsort(torch.rand((n, d), dtype=problem.dtype, device=problem.device), dim=0)
        return (
            torch.rand((n, d), dtype=problem.dtype, device=problem.device) + order.to(problem.dtype)
        ) / n

    @staticmethod
    def _binary_crossover(parent1: torch.Tensor, parent2: torch.Tensor) -> torch.Tensor:
        """Compute binary crossover."""
        exchange = (
            torch.rand(parent1.shape, dtype=torch.get_default_dtype(), device=parent1.device) < 0.5
        )
        first = torch.where(exchange, parent2, parent1)
        second = torch.where(exchange, parent1, parent2)
        return torch.cat([first, second], dim=0)

    @staticmethod
    def _real_crossover(parent1: torch.Tensor, parent2: torch.Tensor) -> torch.Tensor:
        """Compute real crossover."""
        mu = torch.rand_like(parent1)
        beta = torch.where(mu <= 0.5, (2 * mu) ** (1 / 21), (2 - 2 * mu) ** (-1 / 21))
        sign = torch.where(
            torch.randint(0, 2, parent1.shape, device=parent1.device) == 0,
            torch.ones_like(beta),
            -torch.ones_like(beta),
        )
        beta *= sign
        first = (parent1 + parent2) / 2 + beta * (parent1 - parent2) / 2
        second = (parent1 + parent2) / 2 - beta * (parent1 - parent2) / 2
        return torch.cat([first, second], dim=0)

    @staticmethod
    def _real_mutation(offspring: torch.Tensor, problem: Problem) -> torch.Tensor:
        """Compute real mutation."""
        n, d = offspring.shape
        lower = problem.lower.view(1, d).expand(n, d)
        upper = problem.upper.view(1, d).expand(n, d)
        offspring = torch.minimum(torch.maximum(offspring, lower), upper)
        site = torch.rand_like(offspring) < 1 / d
        mu = torch.rand_like(offspring)
        span = upper - lower
        safe_span = torch.where(span != 0, span, torch.ones_like(span))
        left = 1 - (offspring - lower) / safe_span
        right = 1 - (upper - offspring) / safe_span
        lower_delta = (2 * mu + (1 - 2 * mu) * left**21) ** (1 / 21) - 1
        upper_delta = 1 - (2 * (1 - mu) + 2 * (mu - 0.5) * right**21) ** (1 / 21)
        delta = torch.where(mu <= 0.5, lower_delta, upper_delta)
        offspring = torch.where(site, offspring + span * delta, offspring)
        return torch.minimum(torch.maximum(offspring, lower), upper)

    @staticmethod
    def _model_training(
        mask: torch.Tensor, dec: torch.Tensor, real: bool, problem: Problem
    ) -> tuple[_RBM, _DAE | None, torch.Tensor, torch.Tensor]:
        """Compute model training."""
        all_zero = torch.all(~mask, dim=0)
        all_one = torch.all(mask, dim=0)
        other = ~all_zero & ~all_one
        frequency = torch.mean(
            (torch.abs(mask[:, other].to(dec.dtype) * dec[:, other]) > 1e-6).to(dec.dtype),
            dim=0,
        )
        hidden = int(
            torch.sum(
                frequency > torch.rand(frequency.shape, dtype=dec.dtype, device=dec.device)
            ).item()
        )
        hidden = min(max(hidden, 1), mask.shape[0])
        rbm = _RBM(int(other.sum().item()), hidden, problem)
        rbm.train(mask[:, other])
        dae = _DAE(dec.shape[1], hidden, dec.shape[0], problem) if real else None
        if dae is not None:
            dae.train(dec)
        return rbm, dae, all_zero, all_one

    @classmethod
    def _operator(
        cls,
        problem: Problem,
        parent_dec: torch.Tensor,
        parent_mask: torch.Tensor,
        rbm: _RBM | None,
        dae: _DAE | None,
        learned_site: torch.Tensor,
        all_zero: torch.Tensor | None,
        all_one: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute operator."""
        half = parent_mask.shape[0] // 2
        p1_mask = parent_mask[:half]
        p2_mask = parent_mask[half : 2 * half]
        p1_dec = parent_dec[:half]
        p2_dec = parent_dec[half : 2 * half]

        learned_masks: list[torch.Tensor] = []
        learned_decisions: list[torch.Tensor] = []
        if torch.any(learned_site):
            if rbm is None or all_zero is None or all_one is None:
                raise RuntimeError("learned mask generation requires a fitted RBM state")
            other = ~all_zero & ~all_one
            hidden1 = rbm.reduce(p1_mask[learned_site][:, other])
            hidden2 = rbm.reduce(p2_mask[learned_site][:, other])
            recovered = rbm.recover(cls._binary_crossover(hidden1, hidden2))
            learned_mask = torch.zeros(
                (recovered.shape[0], problem.D),
                dtype=torch.bool,
                device=problem.device,
            )
            learned_mask[:, other] = recovered
            learned_mask[:, all_one] = True
            learned_masks.append(learned_mask)
            if torch.any(problem.encoding != 4):
                if dae is None:
                    raise RuntimeError("learned real decisions require a fitted DAE state")
                hidden_dec1 = dae.reduce(p1_dec[learned_site])
                hidden_dec2 = dae.reduce(p2_dec[learned_site])
                learned_decisions.append(dae.recover(cls._real_crossover(hidden_dec1, hidden_dec2)))

        unlearned = ~learned_site
        if torch.any(unlearned):
            learned_masks.append(cls._binary_crossover(p1_mask[unlearned], p2_mask[unlearned]))
            if torch.any(problem.encoding != 4):
                learned_decisions.append(cls._real_crossover(p1_dec[unlearned], p2_dec[unlearned]))

        off_mask = torch.cat(learned_masks, dim=0)
        mutation = (
            torch.rand(off_mask.shape, dtype=problem.dtype, device=problem.device)
            < 1 / off_mask.shape[1]
        )
        off_mask = torch.logical_xor(off_mask, mutation)

        if torch.any(problem.encoding != 4):
            off_dec = cls._real_mutation(torch.cat(learned_decisions, dim=0), problem)
            off_dec[:, problem.encoding == 4] = 1
        else:
            off_dec = torch.ones_like(off_mask, dtype=problem.dtype)
        return off_dec, off_mask

    @staticmethod
    def _environmental_selection(
        population: Population,
        dec: torch.Tensor,
        mask: torch.Tensor,
        n: int,
        old_length: int,
        learned_count: int,
    ) -> tuple[Population, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, float]:
        """Compute environmental selection."""
        success = torch.zeros(
            population.objs.shape[0], dtype=torch.bool, device=population.objs.device
        )
        unique = unique_row_indices(population.objs)
        if unique.numel() == 1:
            unique = unique_row_indices(population.decs)
        unique_population = subset_population(population, unique)
        unique_dec = dec[unique]
        unique_mask = mask[unique]
        n = min(n, unique_population.objs.shape[0])

        front_no, max_front = nd_sort(unique_population.objs, unique_population.cons, n_sort=n)
        crowd_dis = crowding_distance(unique_population.objs, front_no)
        selected = front_no < max_front
        remaining = n - int(selected.sum().item())
        last = torch.where(front_no == max_front)[0]
        rank = torch.argsort(crowd_dis[last], descending=True, stable=True)
        selected[last[rank[:remaining]]] = True

        success[unique[selected]] = True
        learned_success = success[old_length : old_length + learned_count].sum()
        standard_success = success[old_length + learned_count :].sum()
        total_success = (learned_success + standard_success).to(population.objs.dtype)
        ratio = (learned_success.to(population.objs.dtype) + 1e-6) / (total_success + 1e-6)
        ratio_value = float(torch.clamp(ratio, 0.1, 0.9).item())

        index = torch.where(selected)[0]
        return (
            subset_population(unique_population, index),
            unique_dec[index],
            unique_mask[index],
            front_no[index],
            crowd_dis[index],
            ratio_value,
        )

    def _solve(self, problem: Problem, output, should_stop, should_pause) -> None:
        """Execute the vectorized evolutionary loop."""
        torch.set_default_dtype(problem.dtype)
        latin_dec = self._latin(problem.N, problem.D, problem)
        dec = latin_dec * (problem.upper - problem.lower) + problem.lower
        dec[:, problem.encoding == 4] = 1
        mask = self._latin(problem.N, problem.D, problem) > 0.5
        population = problem.evaluation(dec * mask.to(problem.dtype))
        population, dec, mask, front_no, crowd_dis, _ = self._environmental_selection(
            population, dec, mask, problem.N, 0, 0
        )

        rho = 0.5
        half = math.ceil(problem.N / 2)
        while self.not_terminated(problem, population, output, should_stop, should_pause):
            learned_site = rho > torch.rand((half,), dtype=problem.dtype, device=problem.device)
            if torch.any(learned_site):
                rbm, dae, all_zero, all_one = self._model_training(
                    mask, dec, bool(torch.any(problem.encoding != 4).item()), problem
                )
            else:
                rbm = dae = all_zero = all_one = None
            mating = tournament_selection(2, half * 2, front_no, -crowd_dis)
            off_dec, off_mask = self._operator(
                problem,
                dec[mating],
                mask[mating],
                rbm,
                dae,
                learned_site,
                all_zero,
                all_one,
            )
            offspring = problem.evaluation(off_dec * off_mask.to(problem.dtype))
            old_length = population.objs.shape[0]
            learned_count = int(2 * learned_site.sum().item())
            population, dec, mask, front_no, crowd_dis, success_ratio = (
                self._environmental_selection(
                    merge_pop(population, offspring),
                    torch.cat([dec, off_dec], dim=0),
                    torch.cat([mask, off_mask], dim=0),
                    problem.N,
                    old_length,
                    learned_count,
                )
            )
            rho = (rho + success_ratio) / 2
