"""Probability distribution classes for defining parameter spaces in genetic algorithms."""

import math
from collections.abc import Callable, Iterable
from statistics import NormalDist
from typing import Any

import numpy as np

from geneticpy.distributions.distribution_base import DistributionBase


def _sample_truncated(
    cdf: Callable[[float], float], inv_cdf: Callable[[float], float], low: float | None, high: float | None
) -> float:
    """Sample from a distribution truncated to [low, high] using inverse transform sampling."""
    u_low = cdf(low) if low is not None else 0.0
    u_high = cdf(high) if high is not None else 1.0
    if low is not None and high is not None and u_high - u_low < 1e-12:
        # The interval is too far into a tail for the CDF to resolve, so fall back to a uniform draw
        return float(np.random.uniform(low, high))
    # Keep u strictly inside (0, 1), where the inverse CDFs are defined
    u = min(max(np.random.uniform(u_low, u_high), 1e-16), 1 - 1e-16)
    return inv_cdf(u)


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
        """Pull a random value from the Gaussian distribution, truncated to its bounds."""
        normal = NormalDist(self.mean, self.standard_deviation)
        value = _sample_truncated(normal.cdf, normal.inv_cdf, self.low, self.high)
        value = self.constrain(value)
        return self.q_round(value)

    def pull_constrained_value(self, low: float, high: float) -> float:
        """Pull a value from the Gaussian distribution truncated to the low and high bounds."""
        low, high = min(low, high), max(low, high)
        normal = NormalDist(self.mean, self.standard_deviation)
        value = _sample_truncated(normal.cdf, normal.inv_cdf, low, high)
        value = self.constrain(value, low, high)
        return self.q_round(value)


class ChoiceDistribution(DistributionBase):
    """
    Discrete choice distribution for sampling from a list of options.

    Parameters
    ----------
    choice_list : list
        List of possible values to choose from.
    probabilities : str | Iterable[float], optional
        Either "uniform" for uniform probability or a sequence of probabilities (must sum to 1).

    Examples
    --------
    >>> dist = ChoiceDistribution(["add", "multiply", "subtract"])
    >>> value = dist.pull_value()  # Returns one of the three operations
    """

    def __init__(self, choice_list: list, probabilities: str | Iterable[float] = "uniform") -> None:
        assert isinstance(choice_list, list)
        self.choice_list = choice_list
        self.probabilities: list[float] | None = None if isinstance(probabilities, str) else list(probabilities)

    def pull_value(self) -> Any:
        """Pull a random choice from the list."""
        index = np.random.choice(len(self.choice_list), p=self.probabilities)
        return self.choice_list[index]

    def pull_constrained_value(self, low: Any, high: Any) -> Any:
        """Pull a random choice between low and high values."""
        return low if np.random.random() < 0.5 else high


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

    def _cdf(self, x: float) -> float:
        return -math.expm1(-x / self.scale) if x > 0 else 0.0

    def _inv_cdf(self, u: float) -> float:
        return -self.scale * math.log1p(-u)

    def pull_value(self) -> float:
        """Pull a random value from the exponential distribution, truncated to its bounds."""
        value = _sample_truncated(self._cdf, self._inv_cdf, self.low, self.high)
        value = self.constrain(value)
        return self.q_round(value)

    def pull_constrained_value(self, low: float, high: float) -> float:
        """Pull a value from the exponential distribution truncated to the low and high bounds."""
        low, high = min(low, high), max(low, high)
        value = _sample_truncated(self._cdf, self._inv_cdf, low, high)
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

    def _cdf(self, x: float) -> float:
        return NormalDist(self.mean, self.sigma).cdf(math.log(x)) if x > 0 else 0.0

    def _inv_cdf(self, u: float) -> float:
        return math.exp(NormalDist(self.mean, self.sigma).inv_cdf(u))

    def pull_value(self) -> float:
        """Pull a random value from the log-normal distribution, truncated to its bounds."""
        value = _sample_truncated(self._cdf, self._inv_cdf, self.low, self.high)
        value = self.constrain(value)
        return self.q_round(value)

    def pull_constrained_value(self, low: float, high: float) -> float:
        """Pull a value from the log-normal distribution truncated to the low and high bounds."""
        low, high = min(low, high), max(low, high)
        value = _sample_truncated(self._cdf, self._inv_cdf, low, high)
        value = self.constrain(value, low, high)
        return self.q_round(value)
