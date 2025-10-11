"""Main optimization interface for genetic algorithm parameter tuning."""

import random
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from time import time
from typing import Any

import numpy as np
from tqdm import tqdm

from geneticpy.distributions import DistributionBase
from geneticpy.population import Population


@dataclass
class OptimizeResult:
    """A dataclass representing the result of the optimize function."""

    #: The best parameter set found during the optimization.
    best_params: dict[str, Any]

    #: The score of the best parameter set found during the optimization.
    best_score: float

    #: The total time taken to run the optimization, in seconds.
    total_time: float


def optimize(
    fn: Callable[[dict[str, Any]], float | Awaitable[float]],
    param_space: Mapping[str, DistributionBase | Any],
    size: int = 100,
    generation_count: int = 10,
    percentage_to_randomly_spawn: float = 0.1,
    mutate_chance: float = 0.35,
    retain_percentage: float = 0.5,
    maximize_fn: bool = False,
    target: float | None = None,
    verbose: bool = False,
    seed: int | None = None,
    patience: int | None = None,
    use_tournament_selection: bool = False,
    tournament_size: int = 3,
    adaptive_mutation: bool = False,
) -> OptimizeResult:
    """
    Run genetic algorithm optimization over a parameter space.

    The ``optimize`` function is used to run the genetic algorithm over the specified parameter space in an effort to
    minimize (or maximize if ``maximize_fn=True``) the specified loss[reward] function, ``fn(params)``.

    Parameters
    ----------
    fn: callable
        A callable function that can be either synchronous or asynchronous. This function is expected to take a
        dictionary of parameters as input and return a float. (e.g. ``def fn(params: dict) -> float``)
    param_space: Dict[str, DistributionBase]
        A dictionary of parameters to tune. Keys should be a string representing the name of the variable, and values
        should be geneticpy distributions.
    size: int, default = 100
        The number of iterations to attempt with every generation.
    generation_count: int, default = 10
        The number of generations to use during the optimization.
    percentage_to_randomly_spawn: float, default = 0.1
        The percentage of iterations within each generation that will be created with random initial values.
    mutate_chance: float, default = 0.35
        The percentage of iterations within each generation that will be filled with parameters mutated from top
        performing iterations in the previous generation.
    retain_percentage: float, default = 0.5
        The percentage of iterations that will be kept at the end of each generation. The best performing iterations, as
        determined by the ``fn`` function will be kept.
    maximize_fn: bool, default = False
        If ``True``, the specified ``fn`` function will be treated as a reward function, otherwise the ``fn`` function
        will be treated as a loss function.
    target: Optional[float], default = None
        If specified, the algorithm will stop searching once a parameter set resulting in a loss/reward better than or
        equal to the specified value has been found.
    verbose: bool, default = False
        If True, a progress bar will be displayed.
    seed: Optional[int], default = None
        If specified, the random number generators used to generate new parameter sets will be seeded, resulting in a
        deterministic and repeatable result.
    patience: Optional[int], default = None
        Early stopping patience. If specified, optimization stops if no improvement for this many generations.
        Helps prevent unnecessary computation when convergence is reached.
    use_tournament_selection: bool, default = False
        If True, use tournament selection for parent selection during breeding instead of random selection.
        Tournament selection often provides better selection pressure.
    tournament_size: int, default = 3
        Number of individuals competing in each tournament (only used if use_tournament_selection=True).
    adaptive_mutation: bool, default = False
        If True, automatically adjust mutation rate based on progress. Increases mutation when stuck,
        decreases when making steady progress.

    Returns
    -------
    OptimizeResult
        A dataclass containing the optimization results with the following attributes:

        - best_params: The best parameter set found during the optimization.
        - best_score: The score of the best parameter set.
        - total_time: The total time taken to run the optimization, in seconds.

    Examples
    --------
    >>> import geneticpy
    >>> def loss_function(params):
    ...     if params["type"] == "add":
    ...         return params["x"] + params["y"]
    ...     elif params["type"] == "multiply":
    ...         return params["x"] * params["y"]
    >>> param_space = {
    ...     "type": geneticpy.ChoiceDistribution(choice_list=["add", "multiply"]),
    ...     "x": geneticpy.UniformDistribution(low=5, high=10, q=1),
    ...     "y": geneticpy.GaussianDistribution(mean=0, standard_deviation=1, low=-1, high=1),
    ... }
    >>> results = geneticpy.optimize(loss_function, param_space)  # doctest: +SKIP
    >>> print(results)  # doctest: +SKIP
    OptimizeResult(top_params={'type': 'add', 'x': 5, 'y': -0.872345}, top_score=4.127655, total_time=12.34)
    """
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)
    if verbose:
        tqdm_total = int(size * (1 + generation_count * (1 - retain_percentage)))
        t = tqdm(desc="Optimizing parameters", total=tqdm_total)
    else:
        t = None

    start_time = time()

    # Initial mutation parameters (may be adapted)
    current_mutate_chance = mutate_chance
    current_random_spawn = percentage_to_randomly_spawn

    pop = Population(
        fn=fn,
        params=param_space,
        size=size,
        percentage_to_randomly_spawn=current_random_spawn,
        mutate_chance=current_mutate_chance,
        retain_percentage=retain_percentage,
        maximize_fn=maximize_fn,
        tqdm_obj=t,
        target=target,
        use_tournament_selection=use_tournament_selection,
        tournament_size=tournament_size,
    )

    top_score = None
    best_score_ever = None
    generations_without_improvement = 0
    i = 0

    while top_score is None and i < generation_count:
        i += 1
        top_score = pop.evolve()

        # Track best score for early stopping and adaptive parameters
        current_best = pop.get_top_score()

        # Early stopping with patience
        if patience is not None:
            if best_score_ever is None:
                best_score_ever = current_best
            else:
                # Check if we improved
                improved = (maximize_fn and current_best > best_score_ever) or (
                    not maximize_fn and current_best < best_score_ever
                )

                if improved:
                    best_score_ever = current_best
                    generations_without_improvement = 0
                else:
                    generations_without_improvement += 1

                # Stop if no improvement for 'patience' generations
                if generations_without_improvement >= patience:
                    if verbose and t is not None:
                        t.write(f"Early stopping: no improvement for {patience} generations")
                    break

        # Adaptive mutation: increase exploration if stuck, decrease if improving
        if adaptive_mutation:
            if generations_without_improvement > 5:
                # Stuck - increase exploration
                pop.mutate_chance = min(0.8, current_mutate_chance * 1.2)
                pop.percentage_to_randomly_spawn = min(0.3, current_random_spawn * 1.5)
            elif generations_without_improvement < 2:
                # Improving - decrease exploration, increase exploitation
                pop.mutate_chance = max(0.1, current_mutate_chance * 0.9)
                pop.percentage_to_randomly_spawn = max(0.01, current_random_spawn * 0.8)

    if top_score is None:
        pop.get_final_scores()

    top_score = pop.get_top_score()
    top_params = pop.get_top_params()
    total_time = time() - start_time
    if t is not None:
        t.close()
    return OptimizeResult(best_params=top_params, best_score=top_score, total_time=total_time)
