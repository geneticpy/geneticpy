import asyncio
from copy import deepcopy
import random

from geneticpy.distributions.distribution_base import DistributionBase
from geneticpy.parameter_set import ParameterSet


class Population:
    def __init__(self, fn, params, size, percentage_to_randomly_spawn=0.05, mutate_chance=0.25, retain_percentage=0.6,
                 maximize_fn=False, tqdm_obj=None, target=None):
        assert isinstance(params, dict)
        assert int(retain_percentage * size) >= 1
        if asyncio.iscoroutinefunction(fn):
            self.fn = fn
        else:
            async def _fn_async(*args, **kwargs):
                return fn(*args, **kwargs)
            self.fn = _fn_async
        self.params = params
        self.size = size
        self.maximize_fn = maximize_fn
        self.percentage_to_randomly_spawn = percentage_to_randomly_spawn
        self.mutate_chance = mutate_chance
        self.retain_percentage = retain_percentage
        self.tqdm_obj = tqdm_obj
        self.target = target
        self.grades = None
        self.population = [self.create_random_set() for _ in range(self.size)]

    def is_achieved_target(self, score):
        return self.target is not None and ((self.maximize_fn and score > self.target)
                                            or (not self.maximize_fn and score < self.target))

    @staticmethod
    async def _evaluate(individual):
        score = await individual.get_score()
        return score, individual

    async def _grade(self):
        return await asyncio.gather(*[self._evaluate(individual) for individual in self.population])

    def evolve(self):
        graded = asyncio.run(self._grade())
        self.grades = sorted(graded, key=lambda x: x[0], reverse=self.maximize_fn)
        top_score = self.grades[0][0]
        graded = [x[1] for x in self.grades]

        if self.is_achieved_target(top_score):
            self.population = graded
            return top_score
        retained_length = int(len(graded) * self.retain_percentage)
        keep = graded[:retained_length]
        m_count = 0
        s_count = 0
        b_count = 0
        for indiv in keep:
            if self.mutate_chance > random.random():
                new_indiv = deepcopy(indiv)
                m_count += 1
                keep.append(new_indiv.mutate())

        for i in range(int(self.size * self.percentage_to_randomly_spawn)):
            keep.append(self.create_random_set())
            s_count += 1

        if len(keep) > self.size:
            keep = keep[:self.size]

        while len(keep) < self.size:
            set1 = random.randint(0, retained_length - 1)
            set2 = random.randint(0, retained_length - 1)
            if set1 != set2:
                b_count += 1
                keep.append(keep[set1].breed(keep[set2]))
        self.population = keep
        return None

    def get_final_scores(self):
        graded = asyncio.run(self._grade())
        self.grades = sorted(graded, key=lambda x: x[0], reverse=self.maximize_fn)
        graded = [x[1] for x in self.grades]
        self.population = graded

    def get_top_score(self):
        return asyncio.run(self.population[0].get_score())

    def get_top_params(self):
        return self.population[0].get_params()

    def create_random_set(self):
        random_params = {k: v.pull_value() if isinstance(v, DistributionBase) else v for k, v in self.params.items()}
        return ParameterSet(params=random_params, param_space=self.params, fn=self.fn, maximize_fn=self.maximize_fn,
                            tqdm_obj=self.tqdm_obj)
