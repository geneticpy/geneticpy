"""Population management for genetic algorithm evolution."""

from __future__ import annotations

import asyncio
import random
from collections.abc import Awaitable, Callable, Mapping
from copy import deepcopy
from typing import Any

from tqdm import tqdm

from geneticpy.distributions.distribution_base import DistributionBase
from geneticpy.parameter_set import ParameterSet


class Population:
    """
    Manage a population of parameter sets that evolve through genetic operations.

    The population evolves through selection, mutation, and breeding to optimize
    the objective function.

    Parameters
    ----------
    fn : Callable[[dict[str, Any]], float | Awaitable[float]]
        Objective function to optimize (can be sync or async).
    params : Mapping[str, DistributionBase | Any]
        Parameter space defining search distributions.
    size : int
        Number of parameter sets in the population.
    percentage_to_randomly_spawn : float, optional
        Fraction of population to spawn randomly each generation.
    mutate_chance : float, optional
        Probability of mutating a retained parameter set.
    retain_percentage : float, optional
        Fraction of top performers to retain each generation.
    maximize_fn : bool, optional
        Whether to maximize (True) or minimize (False) the objective.
    tqdm_obj : tqdm | None, optional
        Progress bar for tracking evaluations.
    target : float | None, optional
        Target score to stop optimization early if achieved.
    """

    def __init__(
        self,
        fn: Callable[[dict[str, Any]], float | Awaitable[float]],
        params: Mapping[str, DistributionBase | Any],
        size: int,
        percentage_to_randomly_spawn: float = 0.05,
        mutate_chance: float = 0.25,
        retain_percentage: float = 0.6,
        maximize_fn: bool = False,
        tqdm_obj: tqdm | None = None,
        target: float | None = None,
    ) -> None:
        assert isinstance(params, dict)
        assert int(retain_percentage * size) >= 1
        if asyncio.iscoroutinefunction(fn):
            self.fn = fn
        else:

            async def _fn_async(*args: Any, **kwargs: Any) -> float:
                return fn(*args, **kwargs)  # type: ignore[return-value]

            self.fn = _fn_async
        self.params = params
        self.size = size
        self.maximize_fn = maximize_fn
        self.percentage_to_randomly_spawn = percentage_to_randomly_spawn
        self.mutate_chance = mutate_chance
        self.retain_percentage = retain_percentage
        self.tqdm_obj = tqdm_obj
        self.target = target
        self.grades: list[tuple[float, ParameterSet]] | None = None
        self.population: list[ParameterSet] = [self.create_random_set() for _ in range(self.size)]

    def is_achieved_target(self, score: float) -> bool:
        """
        Check if the target score has been achieved.

        Parameters
        ----------
        score : float
            Score to check against target.

        Returns
        -------
        bool
            True if target is achieved, False otherwise.
        """
        return self.target is not None and (
            (self.maximize_fn and score > self.target) or (not self.maximize_fn and score < self.target)
        )

    @staticmethod
    async def _evaluate(individual: ParameterSet) -> tuple[float, ParameterSet]:
        score = await individual.get_score()
        return score, individual

    async def _grade(self) -> list[tuple[float, ParameterSet]]:
        return await asyncio.gather(*[self._evaluate(individual) for individual in self.population])

    def evolve(self) -> float | None:
        """
        Evolve the population by one generation.

        Returns
        -------
        float | None
            Top score if target achieved, None otherwise.
        """
        graded_tuples = asyncio.run(self._grade())
        self.grades = sorted(graded_tuples, key=lambda x: x[0], reverse=self.maximize_fn)
        top_score = self.grades[0][0]
        graded = [x[1] for x in self.grades]

        if self.is_achieved_target(top_score):
            self.population = graded
            return top_score
        retained_length = int(len(graded) * self.retain_percentage)
        keep = graded[:retained_length]
        for indiv in keep:
            if self.mutate_chance > random.random():
                new_indiv = deepcopy(indiv)
                keep.append(new_indiv.mutate())

        for _ in range(int(self.size * self.percentage_to_randomly_spawn)):
            keep.append(self.create_random_set())

        if len(keep) > self.size:
            keep = keep[: self.size]

        while len(keep) < self.size:
            set1 = random.randint(0, retained_length - 1)
            set2 = random.randint(0, retained_length - 1)
            if set1 != set2:
                keep.append(keep[set1].breed(keep[set2]))
        self.population = keep
        return None

    def get_final_scores(self) -> None:
        """Grade and sort the final population by score."""
        graded_tuples = asyncio.run(self._grade())
        self.grades = sorted(graded_tuples, key=lambda x: x[0], reverse=self.maximize_fn)
        graded = [x[1] for x in self.grades]
        self.population = graded

    def get_top_score(self) -> float:
        """
        Get the score of the top parameter set.

        Returns
        -------
        float
            Best fitness score in the population.
        """
        return asyncio.run(self.population[0].get_score())

    def get_top_params(self) -> dict[str, float]:
        """
        Get the parameters of the top parameter set.

        Returns
        -------
        dict[str, float]
            Best parameter dictionary in the population.
        """
        return self.population[0].get_params()

    def create_random_set(self) -> ParameterSet:
        """
        Create a random parameter set from the parameter space.

        Returns
        -------
        ParameterSet
            Newly created random parameter set.
        """
        random_params = {k: v.pull_value() if isinstance(v, DistributionBase) else v for k, v in self.params.items()}
        return ParameterSet(
            params=random_params,
            param_space=self.params,
            fn=self.fn,
            maximize_fn=self.maximize_fn,
            tqdm_obj=self.tqdm_obj,
        )
