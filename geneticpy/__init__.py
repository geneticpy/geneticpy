from geneticpy.distributions import *
from geneticpy.population import Population


def optimize(fn, param_space, size=100, generation_count=10, percentage_to_randomly_spawn=0.05, mutate_chance=0.25,
             retain_percentage=0.6, maximize_fn=False, verbose=True):
    pop = Population(fn=fn, params=param_space, size=size, percentage_to_randomly_spawn=percentage_to_randomly_spawn,
                     mutate_chance=mutate_chance, retain_percentage=retain_percentage,
                     maximize_fn=maximize_fn, verbose=verbose)

    for _ in range(generation_count):
        pop.evolve()
    pop.get_final_scores()

    top_score = pop.get_top_score()
    top_params = pop.get_top_params()
    return top_params, top_score
