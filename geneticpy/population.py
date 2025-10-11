"""Population management for genetic algorithm evolution."""

from __future__ import annotations

import asyncio
import random
from collections.abc import Awaitable, Callable, Mapping
from copy import deepcopy
from typing import Any

from tqdm import tqdm

from geneticpy.distributions import ChoiceDistribution
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
    use_tournament_selection : bool, optional
        If True, use tournament selection for breeding instead of random selection.
    tournament_size : int, optional
        Number of individuals in each tournament (default 3).
    diversity_threshold : float, optional
        Minimum diversity threshold. If diversity drops below this, inject random individuals.
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
        use_tournament_selection: bool = False,
        tournament_size: int = 3,
        diversity_threshold: float = 0.001,
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
        self.use_tournament_selection = use_tournament_selection
        self.tournament_size = tournament_size
        self.diversity_threshold = diversity_threshold
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

    def calculate_diversity(self) -> float:
        """
        Calculate population diversity as normalized variance across all parameters.

        For numeric parameters, uses variance. For choice parameters (categorical),
        uses the ratio of unique values to total possible values.

        Returns
        -------
        float
            Diversity score (0 = no diversity, higher = more diversity).
            Returns 0 if population is empty or has only one individual.
        """
        if len(self.population) < 2:
            return 0.0

        if not self.params:
            return 0.0

        # Calculate diversity for each parameter
        diversity_scores = []

        for key, distribution in self.params.items():
            values = [ind.params[key] for ind in self.population if key in ind.params]
            if len(values) < 2:
                continue

            # For ChoiceDistribution, measure diversity as unique value ratio
            if isinstance(distribution, ChoiceDistribution):
                unique_count = len(set(map(str, values)))  # Convert to str to handle any type
                total_possible = len(distribution.choice_list)
                if total_possible > 0:
                    diversity_scores.append(unique_count / total_possible)
            # For numeric distributions, use normalized variance
            elif isinstance(distribution, DistributionBase):
                try:
                    # Only process if values are numeric
                    numeric_values = [float(v) for v in values]
                    mean = sum(numeric_values) / len(numeric_values)
                    variance = sum((x - mean) ** 2 for x in numeric_values) / len(numeric_values)

                    # Normalize variance by parameter range if available
                    param_range = 1.0
                    if (
                        hasattr(distribution, "low")
                        and hasattr(distribution, "high")
                        and distribution.low is not None
                        and distribution.high is not None
                    ):
                        param_range = distribution.high - distribution.low
                        if param_range > 0:
                            variance = variance / (param_range**2)

                    diversity_scores.append(variance)
                except (ValueError, TypeError):
                    # Skip non-numeric parameters
                    continue

        # Return average diversity across all parameters
        return sum(diversity_scores) / len(diversity_scores) if diversity_scores else 0.0

    def inject_diversity(self, percentage: float = 0.2) -> None:
        """
        Inject diversity by replacing bottom performers with random individuals.

        Parameters
        ----------
        percentage : float, optional
            Fraction of population to replace with random individuals (default 0.2).
        """
        num_to_replace = int(self.size * percentage)
        if num_to_replace > 0:
            # Replace worst performers with random individuals
            for i in range(num_to_replace):
                self.population[-(i + 1)] = self.create_random_set()

    def tournament_select(self, tournament_size: int = 3) -> ParameterSet:
        """
        Select an individual using tournament selection.

        Parameters
        ----------
        tournament_size : int, optional
            Number of individuals to compete in tournament (default 3).

        Returns
        -------
        ParameterSet
            Selected individual from tournament.
        """
        # Ensure tournament size doesn't exceed population
        tournament_size = min(tournament_size, len(self.population))
        contestants = random.sample(self.population, tournament_size)

        # Return best from tournament
        if self.maximize_fn:
            return max(contestants, key=lambda x: x.score if x.score is not None else float("-inf"))
        else:
            return min(contestants, key=lambda x: x.score if x.score is not None else float("inf"))

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

        # Apply mutations to copies of retained individuals
        for indiv in keep:
            if self.mutate_chance > random.random():
                new_indiv = deepcopy(indiv)
                keep.append(new_indiv.mutate())

        # Add random individuals for exploration
        for _ in range(int(self.size * self.percentage_to_randomly_spawn)):
            keep.append(self.create_random_set())

        # Elitism: always keep the best individual at the start
        # Remove excess individuals from the end (not the beginning where elite are)
        if len(keep) > self.size:
            # Keep first retained_length (elite) + fill rest with mutations/breeding
            keep = keep[:retained_length] + keep[retained_length : self.size]

        # Fill remaining slots with breeding
        while len(keep) < self.size:
            if self.use_tournament_selection:
                # Use tournament selection for parent selection
                parent1 = self.tournament_select(self.tournament_size)
                parent2 = self.tournament_select(self.tournament_size)
                keep.append(parent1.breed(parent2))
            else:
                # Original random selection from retained elite
                set1 = random.randint(0, retained_length - 1)
                set2 = random.randint(0, retained_length - 1)
                keep.append(keep[set1].breed(keep[set2]))

        self.population = keep

        # Check diversity and inject random individuals if needed
        diversity = self.calculate_diversity()
        if diversity < self.diversity_threshold:
            self.inject_diversity(percentage=0.1)

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

    def get_top_params(self) -> dict[str, Any]:
        """
        Get the parameters of the top parameter set.

        Returns
        -------
        dict[str, Any]
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
