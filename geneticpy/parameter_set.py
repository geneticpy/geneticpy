import random
from copy import deepcopy


class ParameterSet:

    def __init__(self, params, param_space, fn, maximize_fn):
        self.params = params
        self.param_space = param_space
        self.fn = fn
        self.maximize_fn = maximize_fn
        self.score = None

    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        result.params = deepcopy(self.params)
        result.param_space = self.param_space
        result.fn = self.fn
        result.maximize_fn = self.maximize_fn
        result.score = None
        return result

    def mutate(self):
        self.score = None
        keys = list(self.param_space.keys())
        param = random.choice(keys)
        self.params[param] = self.param_space[param].pull_value()
        return self

    def breed(self, mate):
        child = deepcopy(self)
        child.params = {k: child.param_space[k].pull_constrained_value(v, mate.params[k]) for k, v in child.params.items()}
        return child

    def get_score(self):
        if self.score is None:
            self.score = self.fn(self.params)
        return self.score

    def get_params(self):
        return self.params
