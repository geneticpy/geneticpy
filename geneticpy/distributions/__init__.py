import numpy as np

from geneticpy.distributions.distribution_base import DistributionBase


class UniformDistribution(DistributionBase):
    def __init__(self, low, high, q=None):
        assert low is not None and high is not None
        assert low < high
        assert q is None or q > 0
        self.low = low
        self.high = high
        self.q = q

    def pull_value(self):
        value = np.random.uniform(self.low, self.high)
        return self.q_round(value)

    def pull_constrained_value(self, low, high):
        value = np.random.uniform(low, high)
        return self.q_round(value)


class GaussianDistribution(DistributionBase):
    def __init__(self, mean, standard_deviation, q=None, low=None, high=None):
        assert mean is not None and standard_deviation is not None
        assert standard_deviation > 0
        assert low is None or high is None or (low < high)
        assert q is None or q > 0
        self.mean = mean
        self.standard_deviation = standard_deviation
        self.q = q
        self.low = low
        self.high = high

    def pull_value(self):
        value = np.random.normal(self.mean, self.standard_deviation)
        value = self.constrain(value)
        return self.q_round(value)

    def pull_constrained_value(self, low, high):
        low = min(low, high)
        high = max(low, high)
        constrained_mean = (high + low) / 2
        new_mean = (self.mean + constrained_mean) / 2
        new_standard_deviation = high - low
        value = np.random.normal(new_mean, new_standard_deviation)
        value = self.constrain(value, low, high)
        return self.q_round(value)


class ChoiceDistribution(DistributionBase):
    def __init__(self, choice_list, probabilities='uniform'):
        assert isinstance(choice_list, list)
        self.choice_list = choice_list
        self.probabilities = probabilities

    def pull_value(self):
        return np.random.choice(a=self.choice_list, size=1,
                                p=None if self.probabilities == 'uniform' else self.probabilities)[0]

    def pull_constrained_value(self, low, high):
        """The 'low' and 'high' arguments can be used to represent two possible choices."""
        return np.random.choice(a=[low, high], size=1)[0]


class ExponentialDistribution(DistributionBase):
    def __init__(self, scale=1.0, q=None, low=None, high=None):
        assert scale > 0
        assert q is None or q > 0
        assert low is None or high is None or (low < high)
        assert high is None or high > 0
        self.scale = scale
        self.q = q
        self.low = low
        self.high = high

    def pull_value(self):
        value = np.random.exponential(scale=self.scale, size=None)
        value = self.constrain(value)
        return self.q_round(value)

    def pull_constrained_value(self, low, high):
        value = self.pull_value()
        value = self.constrain(value, low, high)
        return self.q_round(value)


class LogNormalDistribution(DistributionBase):
    def __init__(self, mean=0, sigma=1.0, q=None, low=None, high=None):
        assert sigma > 0
        assert q is None or q > 0
        assert low is None or high is None or (low < high)
        self.mean = mean
        self.sigma = sigma
        self.q = q
        self.low = low
        self.high = high

    def pull_value(self):
        value = np.random.lognormal(mean=self.mean, sigma=self.sigma, size=None)
        value = self.constrain(value)
        return self.q_round(value)

    def pull_constrained_value(self, low, high):
        value = self.pull_value()
        value = self.constrain(value, low, high)
        return self.q_round(value)
