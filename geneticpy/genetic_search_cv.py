from typing import Dict, Generator, Iterable, Optional, Union

import numpy as np
from sklearn.base import BaseEstimator, is_classifier, clone
from sklearn.model_selection import cross_validate
from sklearn.model_selection._split import check_cv
from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import indexable, check_is_fitted, _deprecate_positional_args
from sklearn.utils.metaestimators import if_delegate_has_method

from geneticpy.distributions import DistributionBase
from geneticpy.optimize_function import optimize


class GeneticSearchCV:
    """
    The GeneticSearchCV class can be used as a drop-in replacement for Scikit-Learn's GridSearchCV. This allows for
    faster and more complete optimization of your hyperparameters when using Scikit-Learn estimators and/or pipelines.
    """
    def __init__(self,
                 estimator: BaseEstimator,
                 param_distributions: Dict[str, DistributionBase],
                 *,
                 scoring: Optional[Union[str, callable]] = None,
                 refit: bool = True,
                 cv: Optional[Union[int, Generator, Iterable]] = None,
                 verbose: bool = False,
                 random_state: Optional[int] = None,
                 population_size: int = 50,
                 generation_count: int = 10):
        """
        Parameters
        ----------
        estimator: BaseEstimator
            The estimator that will be used for fitting and predicting subsequently supplied data.
        param_distributions: Dict[str, DistributionBase]
            A dictionary of parameters to tune. Keys should be a string representing the name of the variable, and
            values should be geneticpy distributions.
        scoring: Optional[Union[str, callable]], default = None
            Strategy to evaluate the performance of the cross-validated model on
            the test set.

            If `scoring` represents a single score, one can use:

            - a single string;
            - a callable that returns a single value.
        refit: bool, default = True
            If True, the model will be refit with the best parameters following the hyperparameter tuning.
        cv: Optional[Union[int, Generator, Iterable]], default=None
            Determines the cross-validation splitting strategy.
            Possible inputs for cv are:
            - None, to use the default 5-fold cross validation,
            - integer, to specify the number of folds.
            - `CV splitter`
            - An iterable yielding (train, test) splits as arrays of indices.
        verbose: bool, default = False
            If True, a progress bar will be displayed.
        random_state: Optional[int], default = None
            If specified, the random number generators used to generate new parameter sets will be seeded, resulting in
            a deterministic and repeatable result.
        population_size: int, default = 50
            The number of iterations to attempt with every generation.
        generation_count: int, default = 10
            The number of generations to use during the optimization.

        Examples
        --------
        ::

            from sklearn import datasets
            from sklearn.decomposition import PCA
            from sklearn.linear_model import LogisticRegression
            from sklearn.pipeline import Pipeline

            from geneticpy import GeneticSearchCV, ChoiceDistribution, LogNormalDistribution, UniformDistribution


            # Define a pipeline to search for the best combination of PCA truncation
            # and classifier regularization.
            pca = PCA()
            # set the tolerance to a large value to make the example faster
            logistic = LogisticRegression(max_iter=10000, tol=0.1, solver='saga')
            pipe = Pipeline(steps=[('pca', pca), ('logistic', logistic)])

            X_digits, y_digits = datasets.load_digits(return_X_y=True)

            # Parameters of pipelines can be set using ‘__’ separated parameter names:
            param_grid = {
                'pca__n_components': UniformDistribution(low=5, high=64, q=1),
                'logistic__C': LogNormalDistribution(mean=1, sigma=0.5, low=0.001, high=2),
                'logistic__penalty': ChoiceDistribution(choice_list=['l1', 'l2'])
            }
            search = GeneticSearchCV(pipe, param_grid)
            search.fit(X_digits, y_digits)
            print(f"Best parameter (CV score={search.best_score_}):")
            print(search.best_params_)
        """
        self.estimator = estimator
        self.param_distributions = param_distributions
        self.scoring = scoring
        self.refit = refit
        self.cv = cv
        self.verbose = verbose
        self.random_state = random_state
        self.population_size = population_size
        self.generation_count = generation_count
        if self.random_state is not None:
            np.random.seed(self.random_state)

    @property
    def _estimator_type(self):
        return self.estimator._estimator_type

    @property
    def _pairwise(self):
        # allows cross-validation to see 'precomputed' metrics
        return getattr(self.estimator, '_pairwise', False)

    def score(self, X, y=None):
        """Returns the score on the given data, if the estimator has been refit.
        This uses the score defined by ``scoring`` where provided, and the
        ``best_estimator_.score`` method otherwise.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data, where n_samples is the number of samples and
            n_features is the number of features.
        y : array-like of shape (n_samples, n_output) \
            or (n_samples,), default=None
            Target relative to X for classification or regression;
            None for unsupervised learning.

        Returns
        -------
        score : float
        """
        self._check_is_fitted('score')
        if self.scorer_ is None:
            raise ValueError("No score function explicitly defined, "
                             "and the estimator doesn't provide one %s"
                             % self.best_estimator_)
        score = self.scorer_
        return score(self.best_estimator_, X, y)

    def _check_is_fitted(self, method_name):
        if not self.refit:
            raise NotFittedError('This %s instance was initialized '
                                 'with refit=False. %s is '
                                 'available only after refitting on the best '
                                 'parameters. You can refit an estimator '
                                 'manually using the ``best_params_`` '
                                 'attribute'
                                 % (type(self).__name__, method_name))
        else:
            check_is_fitted(self)

    @if_delegate_has_method(delegate=('best_estimator_', 'estimator'))
    def predict(self, X):
        """Call predict on the estimator with the best found parameters.
        Only available if ``refit=True`` and the underlying estimator supports
        ``predict``.

        Parameters
        ----------
        X : indexable, length n_samples
            Must fulfill the input assumptions of the
            underlying estimator.
        """
        self._check_is_fitted('predict')
        return self.best_estimator_.predict(X)

    @if_delegate_has_method(delegate=('best_estimator_', 'estimator'))
    def predict_proba(self, X):
        """Call predict_proba on the estimator with the best found parameters.
        Only available if ``refit=True`` and the underlying estimator supports
        ``predict_proba``.

        Parameters
        ----------
        X : indexable, length n_samples
            Must fulfill the input assumptions of the
            underlying estimator.
        """
        self._check_is_fitted('predict_proba')
        return self.best_estimator_.predict_proba(X)

    @if_delegate_has_method(delegate=('best_estimator_', 'estimator'))
    def predict_log_proba(self, X):
        """Call predict_log_proba on the estimator with the best found parameters.
        Only available if ``refit=True`` and the underlying estimator supports
        ``predict_log_proba``.

        Parameters
        ----------
        X : indexable, length n_samples
            Must fulfill the input assumptions of the
            underlying estimator.
        """
        self._check_is_fitted('predict_log_proba')
        return self.best_estimator_.predict_log_proba(X)

    @if_delegate_has_method(delegate=('best_estimator_', 'estimator'))
    def decision_function(self, X):
        """Call decision_function on the estimator with the best found parameters.
        Only available if ``refit=True`` and the underlying estimator supports
        ``decision_function``.

        Parameters
        ----------
        X : indexable, length n_samples
            Must fulfill the input assumptions of the
            underlying estimator.
        """
        self._check_is_fitted('decision_function')
        return self.best_estimator_.decision_function(X)

    @if_delegate_has_method(delegate=('best_estimator_', 'estimator'))
    def transform(self, X):
        """Call transform on the estimator with the best found parameters.
        Only available if the underlying estimator supports ``transform`` and
        ``refit=True``.

        Parameters
        ----------
        X : indexable, length n_samples
            Must fulfill the input assumptions of the
            underlying estimator.
        """
        self._check_is_fitted('transform')
        return self.best_estimator_.transform(X)

    @if_delegate_has_method(delegate=('best_estimator_', 'estimator'))
    def inverse_transform(self, Xt):
        """Call inverse_transform on the estimator with the best found params.
        Only available if the underlying estimator implements
        ``inverse_transform`` and ``refit=True``.

        Parameters
        ----------
        Xt : indexable, length n_samples
            Must fulfill the input assumptions of the
            underlying estimator.
        """
        self._check_is_fitted('inverse_transform')
        return self.best_estimator_.inverse_transform(Xt)

    @property
    def n_features_in_(self):
        # For consistency with other estimators we raise a AttributeError so
        # that hasattr() fails if the search estimator isn't fitted.
        try:
            check_is_fitted(self)
        except NotFittedError as nfe:
            raise AttributeError(
                "{} object has no n_features_in_ attribute."
                    .format(self.__class__.__name__)
            ) from nfe

        return self.best_estimator_.n_features_in_

    @property
    def classes_(self):
        self._check_is_fitted("classes_")
        return self.best_estimator_.classes_

    def _run_search(self, X, y, cv):
        def _score(params):
            estimator = clone(self.estimator)
            estimator.set_params(**params)
            scores = cross_validate(estimator, X, y, scoring=self.scoring, cv=cv)
            return scores['test_score'].mean()

        results = optimize(_score,
                           self.param_distributions,
                           maximize_fn=True,
                           seed=self.random_state,
                           verbose=self.verbose,
                           size=self.population_size,
                           generation_count=self.generation_count)
        self.best_params_ = results['top_params']
        self.best_score_ = results['top_score']
        self.best_estimator_ = clone(self.estimator)
        self.best_estimator_.set_params(**self.best_params_)
        if self.refit:
            self.best_estimator_.fit(X, y)

    @_deprecate_positional_args
    def fit(self, X, y=None, *, groups=None, **fit_params):
        """Run fit with all sets of parameters.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training vector, where n_samples is the number of samples and
            n_features is the number of features.
        y : array-like of shape (n_samples, n_output) \
            or (n_samples,), default=None
            Target relative to X for classification or regression;
            None for unsupervised learning.
        groups : array-like of shape (n_samples,), default=None
            Group labels for the samples used while splitting the dataset into
            train/test set. Only used in conjunction with a "Group" :term:`cv`
            instance (e.g., :class:`~sklearn.model_selection.GroupKFold`).
        **fit_params : dict of str -> object
            Parameters passed to the ``fit`` method of the estimator
        """
        estimator = self.estimator
        cv = check_cv(self.cv, y, classifier=is_classifier(estimator))

        X, y, groups = indexable(X, y, groups)

        self._run_search(X, y, cv)

        return self
