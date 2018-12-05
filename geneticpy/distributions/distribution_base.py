from abc import ABC, abstractmethod


class DistributionBase(ABC):

    @abstractmethod
    def __init__(self, low, high, q=None):
        pass

    @abstractmethod
    def pull_value(self):
        pass

    @abstractmethod
    def pull_constrained_value(self, low, high):
        pass

    def q_round(self, value):
        if self.q is not None:
            value = round(value / self.q) * self.q
        return value

    def constrain(self, value, low=None, high=None):
        if low is None:
            low = self.low
        if high is None:
            high = self.high
        if high is not None and value > high:
            value = high
        if low is not None and value < low:
            value = low
        return value
