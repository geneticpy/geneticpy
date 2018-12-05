from abc import ABC, abstractmethod


class DistributionBase(ABC):

    def __init__(self, low, high, q=None):
        pass

    def pull_value(self):
        pass

    def pull_constrained_value(self, low, high):
        pass

    def q_round(self, value):
        if self.q is not None:
            value = round(value / self.q) * self.q
        return value