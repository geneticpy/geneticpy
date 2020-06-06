import pytest

from geneticpy.distributions import *


class TestUniformDistribution:
    def test_uniform_distribution_high_none(self):
        with pytest.raises(AssertionError):
            UniformDistribution(0, None)

    def test_uniform_distribution_low_none(self):
        with pytest.raises(AssertionError):
            UniformDistribution(None, 0)

    def test_uniform_distribution_low_less_than_high(self):
        with pytest.raises(AssertionError):
            UniformDistribution(0, -1)

    def test_uniform_distribution_no_q(self):
        dist = UniformDistribution(0, 1)
        value = dist.pull_value()
        assert value > 0
        assert value < 1

    def test_uniform_distribution_q_float(self):
        dist = UniformDistribution(0, 10, 1.0)
        value = dist.pull_value()
        assert value >= 0
        assert value <= 10
        assert isinstance(value, float)

    def test_uniform_distribution_q_int(self):
        dist = UniformDistribution(0, 10, 1)
        value = dist.pull_value()
        assert value >= 0
        assert value <= 10
        assert isinstance(value, int)

    def test_uniform_distribution_constrained(self):
        dist = UniformDistribution(0, 5)
        value = dist.pull_constrained_value(4.9999, 5)
        assert 4.9999 <= value <= 5


class TestGaussianDistribution:
    def test_gaussian_distribution_no_mean(self):
        with pytest.raises(AssertionError):
            GaussianDistribution(None, 1)

    def test_gaussian_distribution_no_standard_deviation(self):
        with pytest.raises(AssertionError):
            GaussianDistribution(0, None)

    def test_gaussian_distribution_zero_standard_deviation(self):
        with pytest.raises(AssertionError):
            GaussianDistribution(0, 0)

    def test_gaussian_distribution_negative_standard_deviation(self):
        with pytest.raises(AssertionError):
            GaussianDistribution(0, -1)

    def test_gaussian_distribution_low_less_than_high(self):
        with pytest.raises(AssertionError):
            GaussianDistribution(0, 1, low=1, high=0)

    def test_gaussian_distribution_no_q(self):
        dist = GaussianDistribution(5, 0.00001)
        value = dist.pull_value()
        assert 5 == pytest.approx(value, 0.1)

    def test_gaussian_distribution_q_float(self):
        dist = GaussianDistribution(7, 0.00001, q=1.0)
        value = dist.pull_value()
        assert value == 7.0
        assert isinstance(value, float)

    def test_gaussian_distribution_q_int(self):
        dist = GaussianDistribution(7, 0.00001, q=1)
        value = dist.pull_value()
        assert value == 7
        assert isinstance(value, int)

    def test_gaussian_distribution_constrained_q(self):
        dist = GaussianDistribution(-2, 1, q=1)
        value = dist.pull_constrained_value(1, 3)
        assert isinstance(value, int)

    def test_gaussian_distribution_constrained_same(self):
        dist = GaussianDistribution(7, 0.00001)
        value = dist.pull_constrained_value(6.9999, 7.0001)
        assert 7 == pytest.approx(value, 0.1)


class TestChoiceDistribution:
    def test_choice_distribution_no_list(self):
        with pytest.raises(AssertionError):
            ChoiceDistribution('c', 'd')

    def test_choice_distribution_uniform_probability(self):
        dist = ChoiceDistribution(['c', 'd'])
        value = dist.pull_value()
        assert value in ['c', 'd']

    def test_choice_distribution_skewed_probability(self):
        dist = ChoiceDistribution(['a', 'b', 'c', 'd'], [0, 0, 1, 0])
        value = dist.pull_value()
        assert 'c' == value

    def test_choice_distribution_ints(self):
        dist = ChoiceDistribution([-1, 2])
        value = dist.pull_value()
        assert value in [-1, 2]

    def test_choice_distribution_floats(self):
        dist = ChoiceDistribution([-3.0, 5.0])
        value = dist.pull_value()
        assert value in [-3.0, 5.0]

    def test_choice_distribution_constrained(self):
        dist = ChoiceDistribution([x for x in range(99999)])
        value = dist.pull_constrained_value(144, 17711)
        assert value in [144, 17711]


class TestExponentialDistribution:
    def test_exponential_distribution_zero_scale(self):
        with pytest.raises(AssertionError) as e:
            ExponentialDistribution(scale=0)

    def test_exponential_distribution_negative_scale(self):
        with pytest.raises(AssertionError) as e:
            ExponentialDistribution(scale=-1)

    def test_exponential_distribution_low_less_than_high(self):
        with pytest.raises(AssertionError):
            ExponentialDistribution(1, low=2, high=1)

    def test_exponential_distribution_valid_low_and_high(self):
        dist = ExponentialDistribution(.1, low=0.01, high=1)
        value = dist.pull_value()
        assert 0.01 <= value <= 1

    def test_exponential_distribution_valid_q(self):
        dist = ExponentialDistribution(10, q=1)
        value = dist.pull_value()
        assert isinstance(value, int)

    def test_exponential_distribution(self):
        dist = ExponentialDistribution(scale=0.00000001)
        value = dist.pull_value()
        assert 0 < value < 1

    def test_exponential_distribution_constrained(self):
        dist = ExponentialDistribution(scale=25, high=0.05)
        value = dist.pull_value()
        assert value <= 0.05

    def test_exponential_distribution_pull_constrained(self):
        dist = ExponentialDistribution(scale=1)
        value = dist.pull_constrained_value(0.8, 0.81)
        assert 0.8 <= value <= 0.81


class TestLogNormalDistribution:
    def test_log_normal_distribution_zero_sigma(self):
        with pytest.raises(AssertionError):
            LogNormalDistribution(mean=0, sigma=0)

    def test_log_normal_distribution_negative_sigma(self):
        with pytest.raises(AssertionError):
            LogNormalDistribution(mean=0, sigma=-1)

    def test_log_normal_distribution(self):
        dist = LogNormalDistribution(mean=0, sigma=1)
        value = dist.pull_value()
        assert 0 < value

    def test_log_normal_distribution_q(self):
        dist = LogNormalDistribution(mean=0, sigma=1, q=1)
        value = dist.pull_value()
        assert isinstance(value, int)

    def test_log_normal_distribution_high_lower_than_low(self):
        with pytest.raises(AssertionError):
            LogNormalDistribution(mean=0, sigma=1, low=1.1, high=1)

    def test_log_normal_distribution_constrained(self):
        dist = LogNormalDistribution(mean=0, sigma=1, low=15)
        value = dist.pull_value()
        assert value >= 15

    def test_log_normal_distribution_pull_constrained(self):
        dist = LogNormalDistribution(mean=0, sigma=1)
        value = dist.pull_constrained_value(0.8, 0.81)
        assert 0.8 <= value <= 0.81
