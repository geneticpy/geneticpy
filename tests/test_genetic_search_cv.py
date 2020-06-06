import pytest
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from geneticpy import GeneticSearchCV
from geneticpy.distributions import *


def test_logistic_regression():
    iris = load_iris()
    estimator = LogisticRegression(solver='saga',  tol=1e-2, max_iter=200, random_state=0)
    distributions = {
        'C': UniformDistribution(low=1, high=10),
        'penalty': ChoiceDistribution(choice_list=['l1', 'l2'])
    }
    clf = GeneticSearchCV(estimator, distributions, random_state=0)
    search = clf.fit(iris.data, iris.target)
    assert set(search.best_params_.keys()) == {'C', 'penalty'}
    assert 1 <= search.best_params_['C'] <= 10
    assert search.best_params_['penalty'] in ['l1', 'l2']
    best = search.best_estimator_
    assert best.C == search.best_params_['C']
    assert best.penalty == search.best_params_['penalty']


def test_random_forest():
    iris = load_iris()
    estimator = RandomForestClassifier()
    distributions = {
        'n_estimators': UniformDistribution(low=10, high=250, q=1),
        'min_samples_leaf': UniformDistribution(low=1, high=50, q=1)
    }
    clf = GeneticSearchCV(estimator, distributions, random_state=0, generation_count=2, population_size=10)
    search = clf.fit(iris.data, iris.target)
    assert set(search.best_params_.keys()) == {'min_samples_leaf', 'n_estimators'}
    assert 1 <= search.best_params_['min_samples_leaf'] <= 50
    assert 10 <= search.best_params_['n_estimators'] <= 250
    best = search.best_estimator_
    assert search.best_params_['n_estimators'] == best.n_estimators
    assert search.best_params_['min_samples_leaf'] == best.min_samples_leaf


def test_neg_log_loss():
    iris = load_iris()
    estimator = LogisticRegression(solver='saga',  tol=1e-2, max_iter=200, random_state=0)
    distributions = {
        'C': UniformDistribution(low=1, high=10),
        'penalty': ChoiceDistribution(choice_list=['l1', 'l2'])
    }
    clf = GeneticSearchCV(estimator, distributions, random_state=0, scoring='neg_log_loss')
    search = clf.fit(iris.data, iris.target)
    assert set(search.best_params_.keys()) == {'C', 'penalty'}
    assert 1 <= search.best_params_['C'] <= 10
    assert search.best_params_['penalty'] in ['l1', 'l2']
    best = search.best_estimator_
    assert best.C == search.best_params_['C']
    assert best.penalty == search.best_params_['penalty']
