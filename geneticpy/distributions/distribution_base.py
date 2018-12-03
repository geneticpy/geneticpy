from abc import ABC, abstractmethod


class DistributionBase(ABC):

    def __init__(self, low, high, q=None):
        pass

    def pull_value(self):
        pass
