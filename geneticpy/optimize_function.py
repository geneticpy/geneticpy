from time import time
from typing import Any, Dict, Optional, Union

import numpy as np
from tqdm import tqdm

from geneticpy.distributions import DistributionBase
from geneticpy.population import Population


def optimize(fn: callable,
             param_space: Dict[str, DistributionBase],
             size: int = 100,
             generation_count: int = 10,
             percentage_to_randomly_spawn: float = 0.1,
             mutate_chance: float = 0.35,
             retain_percentage: float = 0.5,
             maximize_fn: bool = False,
             target: Optional[float] = None,
             verbose: bool = False,
             seed: Optional[int] = None) -> Dict[str, Union[float, Dict[str, Any]]]:
    """
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

    Returns
    -------
    Dict[str, Union[float, Dict[str, Any]]]:
        A dictionary containing ``top_params``, ``top_score``, and ``total_time`` keys:

        ``top_params``: A dictionary containing the top parameters from the optimization.

        ``top_score``: The score of the ``top_params`` parameter set as determined by the specified ``fn`` function.

        ``total_time``: The total time in seconds that it took to run the optimization.

    Examples
    --------
    ::

        import geneticpy

        def loss_function(params):
            if params['type'] == 'add':
                return params['x'] + params['y']
            elif params['type'] == 'multiply':
                return params['x'] * params['y']

        param_space = {'type': geneticpy.ChoiceDistribution(choice_list=['add', 'multiply']),
                       'x': geneticpy.UniformDistribution(low=5, high=10, q=1),
                       'y': geneticpy.GaussianDistribution(mean=0, standard_deviation=1, low=-1, high=1)}

        results = geneticpy.optimize(loss_function, param_space)
        print(results)

        {'top_params': {'type': 'multiply', 'x': 10, 'y': -1},
         'top_score': -10,
         'total_time': 0.1290111541748047}
    """
    if seed is not None:
        np.random.seed(seed)
    if verbose:
        tqdm_total = int(size * (1 + generation_count * (1 - retain_percentage)))
        t = tqdm(desc='Optimizing parameters', total=tqdm_total)
    else:
        t = None

    start_time = time()
    pop = Population(fn=fn, params=param_space, size=size, percentage_to_randomly_spawn=percentage_to_randomly_spawn,
                     mutate_chance=mutate_chance, retain_percentage=retain_percentage, maximize_fn=maximize_fn,
                     tqdm_obj=t, target=target)

    top_score = None
    i = 0
    while top_score is None and i < generation_count:
        i += 1
        top_score = pop.evolve()
    if top_score is None:
        pop.get_final_scores()

    top_score = pop.get_top_score()
    top_params = pop.get_top_params()
    total_time = time() - start_time
    if t is not None:
        t.close()
    return {
        'top_params': top_params,
        'top_score': top_score,
        'total_time': total_time
    }
