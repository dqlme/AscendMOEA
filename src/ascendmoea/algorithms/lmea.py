"""Implementation helpers for the lmea algorithm."""

from __future__ import annotations

import math
from typing import Optional

import torch

from ..core import Algorithm, Population, Problem
from ..utils import merge_pop, nd_sort, operator_ga, pairwise_dist, tournament_selection
from .sparse_common import convergence, subset_population


class LMEA(Algorithm):
    """Optimize large-scale problems through variable clustering and grouping."""

    name = "LMEA"

    def __init__(
        self,
        save: int = -10,
        n_sel: int = 5,
        n_per: int = 50,
        n_cor: int = 5,
        operator_type: int = 1,
    ):
        """Configure variable-analysis sampling and variation.

        Args:
            save: History retention policy inherited from :class:`Algorithm`.
            n_sel: Population samples used to analyze each variable.
            n_per: Perturbations generated per selected solution.
            n_cor: Trials used in each pairwise correlation test.
            operator_type: ``1`` selects GA and ``2`` selects DE.
        """
        super().__init__(save=save)
        self.n_sel = int(n_sel)
        self.n_per = int(n_per)
        self.n_cor = int(n_cor)
        self.operator_type = int(operator_type)
        if min(self.n_sel, self.n_per, self.n_cor) < 1:
            raise ValueError("n_sel, n_per, and n_cor must be positive")
        if self.operator_type not in {1, 2}:
            raise ValueError("operator_type must be 1 (GA) or 2 (DE)")

    @staticmethod
    def _kmeans_two(points: torch.Tensor) -> torch.Tensor:
        """Partition feature rows into two clusters with batched centroid updates."""
        n = points.shape[0]
        if n < 2:
            return torch.zeros(n, dtype=torch.long, device=points.device)
        first = torch.randint(0, n, (), device=points.device)
        distance = torch.sum((points - points[first]) ** 2, dim=1)
        if torch.sum(distance) > 0:
            second = torch.multinomial(distance / distance.sum(), 1)[0]
        else:
            second = (first + 1) % n
        centroids = torch.stack([points[first], points[second]], dim=0)
        labels = torch.full((n,), -1, dtype=torch.long, device=points.device)
        for _ in range(100):
            new_labels = torch.argmin(pairwise_dist(points, centroids), dim=1)
            if torch.equal(new_labels, labels):
                break
            labels = new_labels
            for cluster in range(2):
                members = labels == cluster
                if torch.any(members):
                    centroids[cluster] = points[members].mean(dim=0)
                else:
                    nearest = torch.min(pairwise_dist(points, centroids), dim=1).values
                    centroids[cluster] = points[torch.argmax(nearest)]
        return labels

    def _variable_clustering(
        self, problem: Problem, population: Population
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Cluster variables by objective sensitivity into position and distance groups."""
        n, d = population.decs.shape
        front_no, _ = nd_sort(population.objs, n_sort=1)
        nondominated = front_no == 1
        fmin = population.objs[nondominated].min(dim=0).values
        fmax = population.objs[nondominated].max(dim=0).values
        if torch.any(fmax == fmin):
            fmin = torch.zeros_like(fmin)
            fmax = torch.ones_like(fmax)

        sample = torch.randint(0, n, (self.n_sel,), device=problem.device)
        base = population.decs[sample].repeat(self.n_per, 1)
        total_perturbations = self.n_sel * self.n_per
        decs = base.view(1, total_perturbations, d).expand(d, -1, -1).clone()
        variable = torch.arange(d, device=problem.device)
        random_value = torch.rand(
            (d, total_perturbations), dtype=problem.dtype, device=problem.device
        )
        random_value = random_value * (problem.upper - problem.lower).unsqueeze(
            1
        ) + problem.lower.unsqueeze(1)
        decs[variable, :, variable] = random_value
        perturbed = problem.evaluation(decs.reshape(-1, d))
        points = perturbed.objs.reshape(d, self.n_per, self.n_sel, problem.M)
        points = points.permute(0, 2, 1, 3)
        points = (points - fmin) / (fmax - fmin)
        points = points - points.mean(dim=2, keepdim=True)

        flat_points = points.reshape(d * self.n_sel, self.n_per, problem.M)
        try:
            _, _, vh = torch.linalg.svd(flat_points, full_matrices=False)
        except RuntimeError:
            _, _, vh_cpu = torch.linalg.svd(flat_points.cpu(), full_matrices=False)
            vh = vh_cpu.to(problem.device)
        vector = vh[:, 0, :]
        vector = vector / torch.linalg.vector_norm(vector, dim=1, keepdim=True).clamp_min(
            torch.finfo(problem.dtype).tiny
        )
        projection = torch.sum(flat_points * vector.unsqueeze(1), dim=2, keepdim=True)
        residual = flat_points - projection * vector.unsqueeze(1)
        rmse = torch.sqrt(torch.sum(residual**2, dim=(1, 2))).reshape(d, self.n_sel)
        normal = torch.ones(problem.M, dtype=problem.dtype, device=problem.device)
        cosine = torch.abs(vector @ normal) / math.sqrt(problem.M)
        angle = torch.rad2deg(torch.acos(torch.clamp(cosine, -1.0, 1.0))).reshape(d, self.n_sel)

        distance_variable = rmse.mean(dim=1) < 1e-2
        cluster = self._kmeans_two(angle)
        labels_in_distance = cluster[distance_variable]
        if torch.any(labels_in_distance == 0) and torch.any(labels_in_distance == 1):
            mean_zero = angle[(cluster == 0) & distance_variable].mean()
            mean_one = angle[(cluster == 1) & distance_variable].mean()
            preferred = torch.where(mean_zero < mean_one, 0, 1)
            distance_variable &= cluster == preferred

        position_variables = torch.where(~distance_variable)[0]
        distance_variables = torch.where(distance_variable)[0]
        return position_variables, distance_variables

    def _correlation_analysis(
        self,
        problem: Problem,
        population: Population,
        distance_variables: torch.Tensor,
    ) -> list[torch.Tensor]:
        """Estimate pairwise variable correlations from controlled perturbations."""
        groups: list[list[int]] = []
        population_size = population.decs.shape[0]
        for variable in distance_variables.detach().cpu().tolist():
            related: list[int] = []
            for group_index, group in enumerate(groups):
                group_related = False
                for other in group:
                    if self._correlation_pair_related(
                        problem,
                        population,
                        population_size,
                        variable,
                        other,
                    ):
                        group_related = True
                        related.append(group_index)
                    if group_related:
                        break
            if not related:
                groups.append([variable])
            else:
                merged: list[int] = []
                for index in related:
                    merged.extend(groups[index])
                merged.append(variable)
                related_set = set(related)
                groups = [group for index, group in enumerate(groups) if index not in related_set]
                groups.append(merged)
        return [torch.tensor(group, dtype=torch.long, device=problem.device) for group in groups]

    @staticmethod
    def _device_rng_state(problem: Problem) -> Optional[torch.Tensor]:
        """Capture the random-number generator state for the active device."""
        if problem.device.type == "cpu":
            return torch.get_rng_state().clone()
        backend = getattr(torch, problem.device.type, None)
        if backend is None or not hasattr(backend, "get_rng_state"):
            return None
        return backend.get_rng_state(problem.device).clone()

    @staticmethod
    def _set_device_rng_state(problem: Problem, state: torch.Tensor) -> None:
        """Restore a previously captured device random-number generator state."""
        if problem.device.type == "cpu":
            torch.set_rng_state(state)
            return
        getattr(torch, problem.device.type).set_rng_state(state, problem.device)

    def _correlation_pair_reference(
        self,
        problem: Problem,
        population: Population,
        population_size: int,
        variable: int,
        other: int,
    ) -> bool:
        """Evaluate paired perturbations used as a correlation reference."""
        for _ in range(self.n_cor):
            parent_index = torch.randint(0, population_size, (), device=problem.device)
            parent_dec = population.decs[parent_index]
            parent_obj = population.objs[parent_index]
            a2 = torch.rand((), dtype=problem.dtype, device=problem.device)
            b2 = torch.rand((), dtype=problem.dtype, device=problem.device)
            a2 = a2 * (problem.upper[variable] - problem.lower[variable]) + problem.lower[variable]
            b2 = b2 * (problem.upper[other] - problem.lower[other]) + problem.lower[other]
            trial = parent_dec.view(1, -1).repeat(3, 1)
            trial[0, variable] = a2
            trial[1, other] = b2
            trial[2, variable] = a2
            trial[2, other] = b2
            evaluated = problem.evaluation(trial)
            delta1 = evaluated.objs[0] - parent_obj
            delta2 = evaluated.objs[2] - evaluated.objs[1]
            if torch.any(delta1 * delta2 < 0):
                return True
        return False

    def _correlation_pair_related(
        self,
        problem: Problem,
        population: Population,
        population_size: int,
        variable: int,
        other: int,
    ) -> bool:
        """Evaluate paired perturbations under the related-variable hypothesis."""
        rng_state = self._device_rng_state(problem)
        if (
            self.n_cor <= 1
            or rng_state is None
            or type(problem).evaluation is not Problem.evaluation
        ):
            return self._correlation_pair_reference(
                problem, population, population_size, variable, other
            )

        parent_indices = []
        a_values = []
        b_values = []
        for _ in range(self.n_cor):
            parent_indices.append(torch.randint(0, population_size, (), device=problem.device))
            a_values.append(torch.rand((), dtype=problem.dtype, device=problem.device))
            b_values.append(torch.rand((), dtype=problem.dtype, device=problem.device))
        parent_indices_t = torch.stack(parent_indices)
        a_values_t = torch.stack(a_values)
        b_values_t = torch.stack(b_values)
        a_values_t = (
            a_values_t * (problem.upper[variable] - problem.lower[variable])
            + problem.lower[variable]
        )
        b_values_t = (
            b_values_t * (problem.upper[other] - problem.lower[other]) + problem.lower[other]
        )

        trials = population.decs[parent_indices_t].repeat_interleave(3, dim=0)
        first = torch.arange(self.n_cor, device=problem.device) * 3
        trials[first, variable] = a_values_t
        trials[first + 1, other] = b_values_t
        trials[first + 2, variable] = a_values_t
        trials[first + 2, other] = b_values_t

        fe_before = problem.FE
        evaluated = problem.evaluation(trials)
        objectives = evaluated.objs.reshape(self.n_cor, 3, problem.M)
        parent_objectives = population.objs[parent_indices_t]
        delta1 = objectives[:, 0] - parent_objectives
        delta2 = objectives[:, 2] - objectives[:, 1]
        related = torch.any(delta1 * delta2 < 0, dim=1)
        related_indices = torch.where(related)[0]
        consumed = int(related_indices[0].item()) + 1 if related_indices.numel() > 0 else self.n_cor

        if consumed < self.n_cor:
            self._set_device_rng_state(problem, rng_state)
            for _ in range(consumed):
                torch.randint(0, population_size, (), device=problem.device)
                torch.rand((), dtype=problem.dtype, device=problem.device)
                torch.rand((), dtype=problem.dtype, device=problem.device)
            problem.FE = fe_before + 3 * consumed
        return consumed < self.n_cor or bool(related[-1].item())

    @staticmethod
    def _replace_rows(
        population: Population, offspring: Population, selected: torch.Tensor
    ) -> Population:
        """Replace selected population rows with offspring tensors in place."""
        return Population(
            torch.where(selected.unsqueeze(1), offspring.decs, population.decs),
            torch.where(selected.unsqueeze(1), offspring.objs, population.objs),
            torch.where(selected.unsqueeze(1), offspring.cons, population.cons),
        )

    def _convergence_optimization(
        self,
        problem: Problem,
        population: Population,
        distance_groups: list[torch.Tensor],
    ) -> Population:
        """Optimize distance-variable groups for objective convergence."""
        n, d = population.decs.shape
        con = convergence(population.objs)
        for group in distance_groups:
            for _ in range(group.numel()):
                mating = tournament_selection(2, 2 * n, con)
                off_dec = population.decs.clone()
                if self.operator_type == 1:
                    new_dec = operator_ga(
                        problem,
                        population.decs[mating],
                        pro_c=1,
                        dis_c=20,
                        pro_m=d / group.numel() / 2,
                        dis_m=20,
                        half=True,
                    )
                else:
                    first = mating[:n]
                    second = mating[n : 2 * n]
                    from ..utils import operator_de

                    new_dec = operator_de(
                        problem,
                        population.decs,
                        population.decs[first],
                        population.decs[second],
                        cr=1,
                        f=0.5,
                        pro_m=d / group.numel() / 2,
                        dis_m=20,
                    )
                off_dec[:, group] = new_dec[:, group]
                offspring = problem.evaluation(off_dec)
                all_con = convergence(torch.cat([population.objs, offspring.objs], dim=0))
                con = all_con[:n]
                new_con = all_con[n:]
                updated = con > new_con
                population = self._replace_rows(population, offspring, updated)
                con = torch.where(updated, new_con, con)
        return population

    @staticmethod
    def _truncation(pop_obj: torch.Tensor, k: int) -> torch.Tensor:
        """Remove crowded candidates by sequential nearest-distance deletion."""
        count = pop_obj.shape[0]
        fmax = pop_obj.max(dim=0).values
        fmin = pop_obj.min(dim=0).values
        span = fmax - fmin
        eps = torch.finfo(pop_obj.dtype).eps
        normalized = (pop_obj - fmin) / span.clamp_min(eps)
        normalized = normalized / torch.linalg.vector_norm(
            normalized, dim=1, keepdim=True
        ).clamp_min(eps)
        cosine = torch.clamp(normalized @ normalized.T, max=1)
        cosine.fill_diagonal_(0)

        chosen = torch.zeros(count, dtype=torch.bool, device=pop_obj.device)
        extreme = torch.argmax(normalized, dim=0)
        chosen[extreme] = True
        chosen_count = int(chosen.sum().item())
        if chosen_count > k:
            selected = torch.where(chosen)[0]
            return selected[torch.randperm(selected.numel(), device=pop_obj.device)[:k]]

        selected = torch.where(chosen)[0]
        nearest = cosine[:, selected].max(dim=1).values
        nearest.masked_fill_(chosen, torch.inf)
        for _ in range(k - chosen_count):
            new_index = torch.argmin(nearest)
            chosen[new_index] = True
            nearest = torch.maximum(nearest, cosine[:, new_index])
            nearest.masked_fill_(chosen, torch.inf)
        return torch.where(chosen)[0]

    @classmethod
    def _environmental_selection(cls, population: Population, n: int) -> Population:
        """Select a diverse nondominated population of the requested size."""
        front_no, max_front = nd_sort(population.objs, n_sort=n)
        selected = front_no < max_front
        last = torch.where(front_no == max_front)[0]
        remaining = n - int(selected.sum().item())
        chosen_last = cls._truncation(population.objs[last], remaining)
        selected[last[chosen_last]] = True
        return subset_population(population, torch.where(selected)[0])

    def _distribution_optimization(
        self,
        problem: Problem,
        population: Population,
        position_variables: torch.Tensor,
    ) -> Population:
        """Optimize position-variable groups to improve front distribution."""
        n = population.decs.shape[0]
        mating = tournament_selection(2, n, convergence(population.objs))
        off_dec = population.decs[mating].clone()
        random_parent = torch.randint(0, n, (n,), device=problem.device)
        new_dec = operator_ga(problem, population.decs[random_parent])
        off_dec[:, position_variables] = new_dec[:, position_variables]
        offspring = problem.evaluation(off_dec)
        return self._environmental_selection(merge_pop(population, offspring), n)

    def _solve(self, problem: Problem, output, should_stop, should_pause) -> None:
        """Execute the vectorized evolutionary loop."""
        torch.set_default_dtype(problem.dtype)
        population = problem.initialization()
        position_variables, distance_variables = self._variable_clustering(problem, population)
        distance_groups = self._correlation_analysis(problem, population, distance_variables)

        while self.not_terminated(problem, population, output, should_stop, should_pause):
            for _ in range(10):
                population = self._convergence_optimization(problem, population, distance_groups)
            for _ in range(problem.M):
                population = self._distribution_optimization(
                    problem, population, position_variables
                )
