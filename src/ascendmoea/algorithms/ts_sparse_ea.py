"""Implementation helpers for the ts_sparse_ea algorithm."""

from __future__ import annotations

import math

import torch

from ..core import Algorithm, Population, Problem
from ..utils import merge_pop, operator_ga, tournament_selection
from .sparse_common import (
    SparsePopulation,
    environmental_selection,
    random_decisions,
    variable_fitness,
)


class TSSparseEA(Algorithm):
    """Perform group-level mask discovery before sparse population evolution."""

    name = "TS-SparseEA"

    def __init__(self, save: int = -10, r_eva: float = 0.1, n_group: int = 50):
        """Configure the initial group-optimization stage.

        Args:
            save: History retention policy inherited from :class:`Algorithm`.
            r_eva: Fraction of the evaluation budget assigned to group search.
            n_group: Number of variables represented by each search group.
        """
        super().__init__(save=save)
        self.r_eva = float(r_eva)
        self.n_group = int(n_group)
        if not 0 < self.r_eva <= 1 or self.n_group < 1:
            raise ValueError("r_eva must be in (0, 1] and n_group must be positive")

    @staticmethod
    def _operator(
        problem: Problem,
        parent_dec: torch.Tensor,
        parent_mask: torch.Tensor,
        real: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Generate offspring values and masks for sparse evolution."""
        half = parent_mask.shape[0] // 2
        parent1 = parent_mask[:half]
        parent2 = parent_mask[half : 2 * half]
        d_mask = parent_mask.shape[1]

        point = torch.randint(1, d_mask + 1, (half, 1), device=problem.device)
        after_point = torch.arange(1, d_mask + 1, device=problem.device).view(1, -1) > point
        off_mask = torch.where(after_point, parent2, parent1)
        mutation = (
            torch.rand(off_mask.shape, dtype=problem.dtype, device=problem.device) < 1 / d_mask
        )
        off_mask = torch.logical_xor(off_mask, mutation)

        if real:
            off_dec = operator_ga(problem, parent_dec, half=True)
        else:
            off_dec = torch.ones((half, problem.D), dtype=problem.dtype, device=problem.device)
        return off_dec, off_mask

    @staticmethod
    def _cosine_similarity(mask: torch.Tensor, normalized_dec: torch.Tensor) -> torch.Tensor:
        """Compare masks and decisions by protected cosine similarity."""
        tiny = torch.finfo(normalized_dec.dtype).tiny
        mask_real = mask.to(normalized_dec.dtype)
        mask_real = mask_real / torch.linalg.vector_norm(
            mask_real, dim=1, keepdim=True
        ).clamp_min(tiny)
        normalized_dec = normalized_dec / torch.linalg.vector_norm(
            normalized_dec, dim=1, keepdim=True
        ).clamp_min(tiny)
        return torch.clamp(mask_real @ normalized_dec.T, min=-1, max=1)

    @classmethod
    def _match_independent(
        cls, dec: torch.Tensor, mask: torch.Tensor, problem: Problem
    ) -> torch.Tensor:
        """Match each source row independently to its nearest candidate."""
        span = problem.upper - problem.lower
        safe_span = span.clamp_min(torch.finfo(problem.dtype).eps)
        normalized = (dec - problem.lower) / safe_span
        similarity = cls._cosine_similarity(mask, normalized)
        selected = torch.argmax(similarity, dim=1)
        matched = normalized[selected]
        return matched * span + problem.lower

    @classmethod
    def _match_one_to_one(
        cls, dec: torch.Tensor, mask: torch.Tensor, problem: Problem
    ) -> torch.Tensor:
        """Construct a one-to-one minimum-distance row matching."""
        span = problem.upper - problem.lower
        safe_span = span.clamp_min(torch.finfo(problem.dtype).eps)
        normalized = (dec - problem.lower) / safe_span
        similarity = cls._cosine_similarity(mask, normalized)
        order = torch.randperm(mask.shape[0], device=problem.device)
        if normalized.device.type == "npu":
            similarity_cpu = similarity.to("cpu")
            order_cpu = order.to("cpu")
            available_cpu = torch.ones(dec.shape[0], dtype=torch.bool)
            selected_by_row_cpu = torch.empty(mask.shape[0], dtype=torch.long)
            for row in order_cpu.tolist():
                score = similarity_cpu[row].masked_fill(~available_cpu, -torch.inf)
                selected = int(torch.argmax(score).item())
                selected_by_row_cpu[row] = selected
                available_cpu[selected] = False
            selected_by_row = selected_by_row_cpu.to(problem.device)
            matched = normalized.index_select(0, selected_by_row)
            return matched * span + problem.lower

        available = torch.ones(dec.shape[0], dtype=torch.bool, device=problem.device)
        matched = torch.zeros_like(mask, dtype=normalized.dtype)
        for row in order.unbind():
            score = similarity[row].masked_fill(~available, -torch.inf)
            selected = torch.argmax(score)
            matched[row] = normalized[selected]
            available[selected] = False
        return matched * span + problem.lower

    def _group_variables(
        self, problem: Problem, real: bool
    ) -> tuple[torch.Tensor, Population, torch.Tensor, torch.Tensor]:
        """Partition decision variables into fitness-balanced groups."""
        fitness, temp_pop, temp_dec, temp_mask = variable_fitness(problem, keep_all_samples=False)
        gamma = int(math.ceil(problem.D / self.n_group))
        order = torch.argsort(fitness, stable=True)
        group_by_rank = torch.div(
            torch.arange(problem.D, device=problem.device),
            gamma,
            rounding_mode="floor",
        ).clamp(max=self.n_group - 1)
        variable_group = torch.empty(problem.D, dtype=torch.long, device=problem.device)
        variable_group[order] = group_by_rank
        group_membership = torch.arange(self.n_group, device=problem.device).view(
            -1, 1
        ) == variable_group.view(1, -1)
        return group_membership, temp_pop, temp_dec, temp_mask

    @staticmethod
    def _expand_group_mask(
        group_mask: torch.Tensor, group_membership: torch.Tensor
    ) -> torch.Tensor:
        """Expand a group-level activation mask to decision columns."""
        dtype = torch.get_default_dtype()
        return (group_mask.to(dtype) @ group_membership.to(dtype)) > 0

    @classmethod
    def _evaluate_group_mask(
        cls,
        problem: Problem,
        group_mask: torch.Tensor,
        group_membership: torch.Tensor,
        reference_dec: torch.Tensor,
    ) -> Population:
        """Evaluate decisions produced by a group-level activation mask."""
        mask = cls._expand_group_mask(group_mask, group_membership)
        return problem.evaluation(reference_dec.view(1, -1) * mask.to(problem.dtype))

    def _binary_group_optimization(
        self,
        problem: Problem,
        state: SparsePopulation,
        real: bool,
    ) -> SparsePopulation:
        """Search group-level masks within the allocated evaluation budget."""
        if real:
            reference = environmental_selection(
                state.population,
                state.dec,
                state.mask,
                problem.M + 1,
                use_decision_uniqueness=True,
            ).mask.to(problem.dtype)
        else:
            reference = torch.ones((1, problem.D), dtype=problem.dtype, device=problem.device)

        group_membership, temp_pop, temp_dec, temp_mask = self._group_variables(problem, real)
        available_evaluations = self.r_eva * problem.max_fe - problem.FE
        generations = math.floor(available_evaluations / problem.N / max(reference.shape[0], 1))

        for reference_dec in reference:
            group_mask = torch.randint(
                0,
                2,
                (problem.N, self.n_group),
                dtype=torch.long,
                device=problem.device,
            ).to(torch.bool)
            group_mask[0] = False
            group_mask[0, 0] = True
            population = self._evaluate_group_mask(
                problem, group_mask, group_membership, reference_dec
            )
            inner = environmental_selection(
                population,
                reference_dec.view(1, -1).expand(problem.N, -1),
                group_mask,
                problem.N,
                use_decision_uniqueness=True,
            )

            for _ in range(max(generations, 0)):
                mating = tournament_selection(2, problem.N, inner.front_no, -inner.crowd_dis)
                repeated_dec = reference_dec.view(1, -1).expand(problem.N, -1)
                _, off_group_mask = self._operator(
                    problem,
                    repeated_dec,
                    inner.mask[mating],
                    real,
                )
                offspring = self._evaluate_group_mask(
                    problem, off_group_mask, group_membership, reference_dec
                )
                merged_count = inner.population.objs.shape[0] + offspring.objs.shape[0]
                inner = environmental_selection(
                    merge_pop(inner.population, offspring),
                    reference_dec.view(1, -1).expand(merged_count, -1),
                    torch.cat([inner.mask, off_group_mask], dim=0),
                    problem.N,
                    use_decision_uniqueness=True,
                )

            off_mask = self._expand_group_mask(inner.mask, group_membership)
            if real:
                off_dec = self._match_independent(state.dec, off_mask, problem)
            else:
                off_dec = torch.ones_like(off_mask, dtype=problem.dtype)
            offspring = problem.evaluation(off_dec * off_mask.to(problem.dtype))
            state = environmental_selection(
                merge_pop(state.population, offspring, temp_pop),
                torch.cat([state.dec, off_dec, temp_dec], dim=0),
                torch.cat([state.mask, off_mask, temp_mask], dim=0),
                problem.N,
                use_decision_uniqueness=True,
            )
        return state

    def _solve(self, problem: Problem, output, should_stop, should_pause) -> None:
        """Execute the vectorized evolutionary loop."""
        torch.set_default_dtype(problem.dtype)
        real = bool((problem.encoding[0] != 4).item())
        dec = (
            random_decisions(problem, problem.N)
            if real
            else torch.ones((problem.N, problem.D), dtype=problem.dtype, device=problem.device)
        )
        mask = torch.rand((problem.N, problem.D), dtype=problem.dtype, device=problem.device) < 0.5
        population = problem.evaluation(dec * mask.to(problem.dtype))
        state = SparsePopulation(
            population,
            dec,
            mask,
            torch.empty(0, dtype=problem.dtype, device=problem.device),
            torch.empty(0, dtype=problem.dtype, device=problem.device),
        )
        state = self._binary_group_optimization(problem, state, real)
        state = environmental_selection(
            state.population,
            state.dec,
            state.mask,
            problem.N,
            use_decision_uniqueness=True,
        )

        while self.not_terminated(problem, state.population, output, should_stop, should_pause):
            mating = tournament_selection(2, problem.N, state.front_no, -state.crowd_dis)
            off_dec, off_mask = self._operator(problem, state.dec[mating], state.mask[mating], real)
            if real:
                off_dec = self._match_one_to_one(off_dec, off_mask, problem)
            offspring = problem.evaluation(off_dec * off_mask.to(problem.dtype))
            state = environmental_selection(
                merge_pop(state.population, offspring),
                torch.cat([state.dec, off_dec], dim=0),
                torch.cat([state.mask, off_mask], dim=0),
                problem.N,
                use_decision_uniqueness=True,
            )
