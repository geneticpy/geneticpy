from time import time

from tqdm import tqdm

from geneticpy.distributions import *
from geneticpy.population import Population


def optimize(fn,
             param_space,
             size=100,
             generation_count=10,
             percentage_to_randomly_spawn=0.1,
             mutate_chance=0.35,
             retain_percentage=0.5,
             maximize_fn=False,
             target=None,
             verbose=False,
             seed=None):
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
