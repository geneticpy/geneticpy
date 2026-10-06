"""Base class for probability distributions used in genetic algorithm parameter spaces."""

import math
from abc import ABC, abstractmethod


class DistributionBase(ABC):
    """
    Abstract base class for parameter distributions.

    All distribution classes must implement pull_value() and pull_constrained_value()
    methods to support both random initialization and constrained breeding.
    """

    @abstractmethod
    def pull_value(self) -> float:
        """
        Pull a random value from the distribution.

        Returns
        -------
        float
            A randomly sampled value from the distribution.
        """
        pass

    @abstractmethod
    def pull_constrained_value(self, low: float, high: float) -> float:
        """
        Pull a value constrained between low and high bounds.

        Parameters
        ----------
        low : float
            Lower bound for the value.
        high : float
            Upper bound for the value.

        Returns
        -------
        float
            A randomly sampled value within the specified bounds.
        """
        pass

    def q_round(self, value: float) -> float:
        """
        Round value to nearest quantization step if q is defined.

        The result is kept within the instance's low and high bounds, if available.

        Parameters
        ----------
        value : float
            The value to quantize.

        Returns
        -------
        float
            The quantized value, or the original value if no quantization is set.
        """
        q = getattr(self, "q", None)
        if q is not None:
            step = round(value / q)
            # Rounding can step past a bound; clamp to the nearest in-bounds step. The small tolerance keeps
            # bounds that are themselves multiples of q (e.g. 0.3 with q=0.1) reachable despite float error.
            low = getattr(self, "low", None)
            high = getattr(self, "high", None)
            if low is not None:
                step = max(step, math.ceil(low / q - 1e-9))
            if high is not None:
                step = min(step, math.floor(high / q + 1e-9))
            value = step * q
        return value

    def constrain(self, value: float, low: float | None = None, high: float | None = None) -> float:
        """
        Constrain value within low and high bounds.

        Parameters
        ----------
        value : float
            The value to constrain.
        low : float | None, optional
            Lower bound. If None, uses instance's low attribute if available.
        high : float | None, optional
            Upper bound. If None, uses instance's high attribute if available.

        Returns
        -------
        float
            The constrained value.
        """
        if low is None:
            low = getattr(self, "low", None)
        if high is None:
            high = getattr(self, "high", None)
        if high is not None and value > high:
            value = high
        if low is not None and value < low:
            value = low
        return value
