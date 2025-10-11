"""Probability distribution classes for defining parameter spaces in genetic algorithms."""

import numpy as np

from geneticpy.distributions.distribution_base import DistributionBase


class UniformDistribution(DistributionBase):
    """
    Uniform distribution for sampling values uniformly within a range.

    Parameters
    ----------
    low : float
        Lower bound of the distribution (inclusive).
    high : float
        Upper bound of the distribution (inclusive).
    q : float | None, optional
        Quantization step size. If specified, sampled values are rounded to nearest multiple of q.

    Examples
    --------
    >>> dist = UniformDistribution(0, 10, q=1)
    >>> value = dist.pull_value()  # Returns integer between 0 and 10
    """

    def __init__(self, low: float, high: float, q: float | None = None) -> None:
        assert low is not None and high is not None
        assert low < high
        assert q is None or q > 0
        self.low = low
        self.high = high
        self.q = q

    def pull_value(self) -> float:
        """Pull a random value uniformly from the distribution."""
        value = np.random.uniform(self.low, self.high)
        return self.q_round(value)

    def pull_constrained_value(self, low: float, high: float) -> float:
        """Pull a random value uniformly within specified bounds."""
        value = np.random.uniform(low, high)
        return self.q_round(value)


class GaussianDistribution(DistributionBase):
    """
    Gaussian (normal) distribution for sampling values around a mean.

    Parameters
    ----------
    mean : float
        Mean of the distribution.
    standard_deviation : float
        Standard deviation of the distribution (must be positive).
    q : float | None, optional
        Quantization step size.
    low : float | None, optional
        Lower bound constraint.
    high : float | None, optional
        Upper bound constraint.

    Examples
    --------
    >>> dist = GaussianDistribution(mean=0, standard_deviation=1, low=-2, high=2)
    >>> value = dist.pull_value()  # Returns value approximately near 0, constrained to [-2, 2]
    """

    def __init__(
        self,
        mean: float,
        standard_deviation: float,
        q: float | None = None,
        low: float | None = None,
        high: float | None = None,
    ) -> None:
        assert mean is not None and standard_deviation is not None
        assert standard_deviation > 0
        assert low is None or high is None or (low < high)
        assert q is None or q > 0
        self.mean = mean
        self.standard_deviation = standard_deviation
        self.q = q
        self.low = low
        self.high = high

    def pull_value(self) -> float:
        """Pull a random value from the Gaussian distribution."""
        value = np.random.normal(self.mean, self.standard_deviation)
        value = self.constrain(value)
        return self.q_round(value)

    def pull_constrained_value(self, low: float, high: float) -> float:
        """Pull a value from Gaussian centered between low and high bounds."""
        low = min(low, high)
        high = max(low, high)
        constrained_mean = (high + low) / 2
        new_mean = (self.mean + constrained_mean) / 2
        new_standard_deviation = high - low
        value = np.random.normal(new_mean, new_standard_deviation)
        value = self.constrain(value, low, high)
        return self.q_round(value)


class ChoiceDistribution(DistributionBase):
    """
    Discrete choice distribution for sampling from a list of options.

    Parameters
    ----------
    choice_list : list
        List of possible values to choose from.
    probabilities : str | list[float], optional
        Either "uniform" for uniform probability or a list of probabilities (must sum to 1).

    Examples
    --------
    >>> dist = ChoiceDistribution(["add", "multiply", "subtract"])
    >>> value = dist.pull_value()  # Returns one of the three operations
    """

    def __init__(self, choice_list: list, probabilities: str | list[float] = "uniform") -> None:
        assert isinstance(choice_list, list)
        self.choice_list = choice_list
        self.probabilities: list[float] | None = (
            None if probabilities == "uniform" else (probabilities if isinstance(probabilities, list) else None)
        )

    def pull_value(self) -> float:
        """Pull a random choice from the list."""
        return np.random.choice(a=self.choice_list, size=1, p=self.probabilities)[0]

    def pull_constrained_value(self, low: float, high: float) -> float:
        """Pull a random choice between low and high values."""
        return np.random.choice(a=[low, high], size=1)[0]


class ExponentialDistribution(DistributionBase):
    """
    Exponential distribution for sampling positive values with exponential decay.

    Parameters
    ----------
    scale : float, optional
        Scale parameter (1/lambda). Higher scale means higher average values.
    q : float | None, optional
        Quantization step size.
    low : float | None, optional
        Lower bound constraint.
    high : float | None, optional
        Upper bound constraint.

    Examples
    --------
    >>> dist = ExponentialDistribution(scale=2.0, low=0, high=10)
    >>> value = dist.pull_value()  # Returns positive value with exponential distribution
    """

    def __init__(
        self, scale: float = 1.0, q: float | None = None, low: float | None = None, high: float | None = None
    ) -> None:
        assert scale > 0
        assert q is None or q > 0
        assert low is None or high is None or (low < high)
        assert high is None or high > 0
        self.scale = scale
        self.q = q
        self.low = low
        self.high = high

    def pull_value(self) -> float:
        """Pull a random value from the exponential distribution."""
        value = np.random.exponential(scale=self.scale, size=None)
        value = self.constrain(value)
        return self.q_round(value)

    def pull_constrained_value(self, low: float, high: float) -> float:
        """Pull a value from exponential distribution constrained to bounds."""
        value = self.pull_value()
        value = self.constrain(value, low, high)
        return self.q_round(value)


class LogNormalDistribution(DistributionBase):
    """
    Log-normal distribution for sampling positive values with log-normal distribution.

    Parameters
    ----------
    mean : float, optional
        Mean of the underlying normal distribution.
    sigma : float, optional
        Standard deviation of the underlying normal distribution (must be positive).
    q : float | None, optional
        Quantization step size.
    low : float | None, optional
        Lower bound constraint.
    high : float | None, optional
        Upper bound constraint.

    Examples
    --------
    >>> dist = LogNormalDistribution(mean=0, sigma=1.0, low=0.1, high=100)
    >>> value = dist.pull_value()  # Returns positive value with log-normal distribution
    """

    def __init__(
        self,
        mean: float = 0,
        sigma: float = 1.0,
        q: float | None = None,
        low: float | None = None,
        high: float | None = None,
    ) -> None:
        assert sigma > 0
        assert q is None or q > 0
        assert low is None or high is None or (low < high)
        self.mean = mean
        self.sigma = sigma
        self.q = q
        self.low = low
        self.high = high

    def pull_value(self) -> float:
        """Pull a random value from the log-normal distribution."""
        value = np.random.lognormal(mean=self.mean, sigma=self.sigma, size=None)
        value = self.constrain(value)
        return self.q_round(value)

    def pull_constrained_value(self, low: float, high: float) -> float:
        """Pull a value from log-normal distribution constrained to bounds."""
        value = self.pull_value()
        value = self.constrain(value, low, high)
        return self.q_round(value)
