from sklearn.datasets import load_iris, load_digits
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

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


def test_pipeline():
    pca = PCA()
    logistic = LogisticRegression(max_iter=10000, tol=0.1)
    pipe = Pipeline(steps=[('pca', pca), ('logistic', logistic)])

    X_digits, y_digits = load_digits(return_X_y=True)

    param_grid = {
        'pca__n_components': UniformDistribution(low=5, high=64, q=1),
        'logistic__C': LogNormalDistribution(mean=1, sigma=0.5, low=0.001, high=2),
        'logistic__penalty': ChoiceDistribution(['l2'])
    }
    search = GeneticSearchCV(pipe, param_grid, random_state=0, generation_count=2, population_size=10)
    search.fit(X_digits, y_digits)
    assert set(search.best_params_.keys()) == {'pca__n_components', 'logistic__C', 'logistic__penalty'}
    assert search.best_params_['logistic__penalty'] == 'l2'
    assert search._estimator_type == 'classifier'
    assert list(search.classes_) == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    assert search.n_features_in_ == 64
    prediction = search.predict(X_digits[0,:].reshape(1, -1))[0]
    assert prediction == 0
    probabilities = search.predict_proba(X_digits[0,:].reshape(1, -1))[0]
    assert 0.5 < probabilities[0] < 1
    for i in range(1, 9):
        assert 0 < probabilities[i] < 0.5


def test_complex_pipeline():
    pca = PCA()
    logistic = LogisticRegression(max_iter=10000, tol=0.1)
    preprocessing = Pipeline(steps=[('pca', pca)])
    pipe = Pipeline(steps=[('preprocessing', preprocessing), ('logistic', logistic)])

    X_digits, y_digits = load_digits(return_X_y=True)

    param_grid = {
        'preprocessing__pca__n_components': UniformDistribution(low=5, high=64, q=1),
        'logistic__C': LogNormalDistribution(mean=1, sigma=0.5, low=0.001, high=2),
        'logistic__penalty': ChoiceDistribution(['l2'])
    }
    search = GeneticSearchCV(pipe, param_grid, random_state=0, generation_count=2, population_size=10)
    search.fit(X_digits, y_digits)
    assert set(search.best_params_.keys()) == {'preprocessing__pca__n_components', 'logistic__C', 'logistic__penalty'}
    assert search.best_params_['logistic__penalty'] == 'l2'
    assert search._estimator_type == 'classifier'
    assert list(search.classes_) == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    assert search.n_features_in_ == 64
    prediction = search.predict(X_digits[0, :].reshape(1, -1))[0]
    assert prediction == 0
    probabilities = search.predict_proba(X_digits[0, :].reshape(1, -1))[0]
    assert 0.5 < probabilities[0] < 1
    for i in range(1, 9):
        assert 0 < probabilities[i] < 0.5
