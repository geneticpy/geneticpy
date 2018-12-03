from geneticpy.distributions import UniformDistribution, GaussianDistribution, ChoiceDistribution
import pytest


def test_uniform_distribution_high_none():
    with pytest.raises(AssertionError):
        dist = UniformDistribution(0, None)


def test_uniform_distribution_low_none():
    with pytest.raises(AssertionError):
        dist = UniformDistribution(None, 0)


def test_uniform_distribution_low_less_than_high():
    with pytest.raises(AssertionError):
        dist = UniformDistribution(0, -1)


def test_uniform_distribution_no_q():
    dist = UniformDistribution(0, 1)
    value = dist.pull_value()
    assert value > 0
    assert value < 1


def test_uniform_distribution_q_float():
    dist = UniformDistribution(0, 10, 1.0)
    value = dist.pull_value()
    assert value >= 0
    assert value <= 10
    assert isinstance(value, float)


def test_uniform_distribution_q_int():
    dist = UniformDistribution(0, 10, 1)
    value = dist.pull_value()
    assert value >= 0
    assert value <= 10
    assert isinstance(value, int)
