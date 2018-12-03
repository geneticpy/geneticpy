from geneticpy.distributions.distribution_base import DistributionBase
import numpy as np


class UniformDistribution(DistributionBase):
    def __init__(self, low, high, q=None):
        assert low is not None and high is not None
        assert low < high
        self.low = low
        self.high = high
        self.q = q

    def pull_value(self):
        value = np.random.uniform(self.low, self.high)
        if self.q is not None:
            value = round(value / self.q) * self.q
        return value


class GaussianDistribution(DistributionBase):
    def __init__(self, mean, standard_deviation, q=None, low=None, high=None):
        assert low is None or high is None or (low < high)
        self.mean = mean
        self.standard_deviation = standard_deviation
        self.q = q
        self.low = low
        self.high = high

    def pull_value(self):
        value = np.random.normal(self.mean, self.standard_deviation)
        if self.low is not None and value < self.low:
            return self.pull_value()  # Outside of range, try again
        if self.high is not None and value > self.high:
            return self.pull_value()  # Outside of range, try again
        if self.q is not None:
            value = round(value / self.q) * self.q
        return value


class ChoiceDistribution(DistributionBase):
    def __init__(self, choice_list, probabilities='uniform'):
        self.choice_list = choice_list
        self.probabilities = probabilities

    def pull_value(self):
        return np.random.choice(a=self.choice_list, size=1,
                                p=None if self.probabilities == 'uniform' else self.probabilities)
