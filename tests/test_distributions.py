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


def test_gaussian_distribution_no_mean():
    with pytest.raises(AssertionError):
        dist = GaussianDistribution(None, 1)


def test_gaussian_distribution_no_standard_deviation():
    with pytest.raises(AssertionError):
        dist = GaussianDistribution(0, None)


def test_gaussian_distribution_zero_standard_deviation():
    with pytest.raises(AssertionError):
        dist = GaussianDistribution(0, 0)


def test_gaussian_distribution_negative_standard_deviation():
    with pytest.raises(AssertionError):
        dist = GaussianDistribution(0, -1)


def test_gaussian_distribution_low_less_than_high():
    with pytest.raises(AssertionError):
        dist = GaussianDistribution(0, 1, low=1, high=0)


def test_gaussian_distribution_no_q():
    dist = GaussianDistribution(5, 0.00001)
    value = dist.pull_value()
    assert 5 == pytest.approx(value, 0.1)


def test_gaussian_distribution_q_float():
    dist = GaussianDistribution(7, 0.00001, q=1.0)
    value = dist.pull_value()
    assert value == 7.0
    assert isinstance(value, float)


def test_gaussian_distribution_q_int():
    dist = GaussianDistribution(7, 0.00001, q=1)
    value = dist.pull_value()
    assert value == 7
    assert isinstance(value, int)


def test_choice_distribution_no_list():
    with pytest.raises(AssertionError):
        dist = ChoiceDistribution('c', 'd')


def test_choice_distribution_uniform_probability():
    dist = ChoiceDistribution(['c', 'd'])
    value = dist.pull_value()
    assert value in ['c', 'd']


def test_choice_distribution_skewed_probability():
    dist = ChoiceDistribution(['a', 'b', 'c', 'd'], [0, 0, 1, 0])
    value = dist.pull_value()
    assert 'c' == value


def test_choice_distribution_ints():
    dist = ChoiceDistribution([-1, 2])
    value = dist.pull_value()
    assert value in [-1, 2]


def test_choice_distribution_floats():
    dist = ChoiceDistribution([-3.0, 5.0])
    value = dist.pull_value()
    assert value in [-3.0, 5.0]
