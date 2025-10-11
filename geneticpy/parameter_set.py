"""Parameter set representation for genetic algorithm individuals."""

from __future__ import annotations

import random
from collections.abc import Awaitable, Callable
from copy import deepcopy
from typing import Any

from tqdm import tqdm

from geneticpy.distributions import DistributionBase


class ParameterSet:
    """
    Represent a single parameter set (individual) in the genetic algorithm population.

    Each parameter set contains a dictionary of parameter values and can mutate,
    breed with other parameter sets, and evaluate its fitness score.

    Parameters
    ----------
    params : dict[str, Any]
        Dictionary of parameter names to values.
    param_space : dict[str, DistributionBase]
        Dictionary of parameter distributions defining the search space.
    fn : Callable[[dict[str, Any]], Awaitable[float]]
        Async objective function to evaluate parameter sets.
    maximize_fn : bool
        Whether to maximize (True) or minimize (False) the objective function.
    tqdm_obj : tqdm | None
        Progress bar object for tracking evaluations.
    """

    def __init__(
        self,
        params: dict[str, Any],
        param_space: dict[str, DistributionBase],
        fn: Callable[[dict[str, Any]], Awaitable[float]],
        maximize_fn: bool,
        tqdm_obj: tqdm | None,
    ) -> None:
        self.params = params
        self.param_space = param_space
        self.fn = fn
        self.maximize_fn = maximize_fn
        self.tqdm_obj = tqdm_obj
        self.score: float | None = None

    def __deepcopy__(self, memo: dict[int, Any]) -> ParameterSet:
        """Create a deep copy of the parameter set for breeding operations."""
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        result.params = deepcopy(self.params)
        result.param_space = self.param_space
        result.fn = self.fn
        result.maximize_fn = self.maximize_fn
        result.score = None
        result.tqdm_obj = self.tqdm_obj
        return result

    def mutate(self, mutation_rate: float = 1.0) -> ParameterSet:
        """
        Mutate the parameter set by randomly changing one or more parameters.

        Parameters
        ----------
        mutation_rate : float, optional
            Controls mutation intensity. If 1.0 (default), mutates exactly one parameter
            (legacy behavior). If < 1.0, represents probability of mutating each parameter.
            For example, 0.5 means each parameter has 50% chance of mutating.

        Returns
        -------
        ParameterSet
            Self, after mutation.
        """
        self.score = None
        keys = [k for k, v in self.param_space.items() if isinstance(v, DistributionBase)]

        if mutation_rate >= 1.0:
            # Legacy behavior: mutate exactly one parameter
            param = random.choice(keys)
            self.params[param] = self.param_space[param].pull_value()
        else:
            # New behavior: probabilistic mutation of each parameter
            mutated = False
            for param in keys:
                if random.random() < mutation_rate:
                    self.params[param] = self.param_space[param].pull_value()
                    mutated = True
            # Ensure at least one mutation occurred
            if not mutated and keys:
                param = random.choice(keys)
                self.params[param] = self.param_space[param].pull_value()

        return self

    def breed(self, mate: ParameterSet) -> ParameterSet:
        """
        Breed with another parameter set to create a child.

        Parameters
        ----------
        mate : ParameterSet
            Another parameter set to breed with.

        Returns
        -------
        ParameterSet
            A new child parameter set with values between parent values.
        """
        child = deepcopy(self)
        child.params = {
            k: (
                child.param_space[k].pull_constrained_value(v, mate.params[k])
                if k in child.param_space and isinstance(child.param_space[k], DistributionBase)
                else v
            )
            for k, v in child.params.items()
        }
        return child

    async def get_score(self) -> float:
        """
        Evaluate and return the fitness score of this parameter set.

        Returns
        -------
        float
            The fitness score from the objective function.

        Raises
        ------
        ValueError
            If the objective function returns None.
        """
        if self.score is None:
            self.score = await self.fn(self.params)
            if self.score is None:
                raise ValueError("Loss function returned None.")
            if self.tqdm_obj is not None:
                self.tqdm_obj.update()
        return self.score

    def get_params(self) -> dict[str, Any]:
        """
        Get the parameter dictionary.

        Returns
        -------
        dict[str, Any]
            Dictionary of parameter names to values.
        """
        return self.params
