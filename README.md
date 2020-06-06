# GeneticPy

[![Build Status](https://travis-ci.com/geneticpy/geneticpy.svg?branch=master)](https://travis-ci.com/geneticpy/geneticpy)
[![codecov](https://codecov.io/gh/geneticpy/geneticpy/branch/master/graph/badge.svg)](https://codecov.io/gh/geneticpy/geneticpy)
[![PyPI version](https://badge.fury.io/py/geneticpy.svg)](https://badge.fury.io/py/geneticpy)
[![PyPI pyversions](https://img.shields.io/pypi/pyversions/geneticpy.svg)](https://pypi.python.org/pypi/geneticpy/)
[![Downloads](https://pepy.tech/badge/geneticpy/week)](https://pepy.tech/project/geneticpy)

GeneticPy is an optimizer that uses a genetic algorithm to quickly search through custom parameter spaces for optimal solutions.

### Installation

GeneticPy requires Python 3.6+

```sh
pip install geneticpy
```

### Optimize Example:

A brief example to get you started is included below:

```python
import geneticpy

def loss_function(params):
  if params['type'] == 'add':
    return params['x'] + params['y']
  elif params['type'] == 'multiply':
    return params['x'] * params['y']

param_space = {'type': geneticpy.ChoiceDistribution(choice_list=['add', 'multiply']),
               'x': geneticpy.UniformDistribution(low=5, high=10, q=1),
               'y': geneticpy.GaussianDistribution(mean=0, standard_deviation=1)}

results = geneticpy.optimize(loss_function, param_space, size=200, generation_count=500, verbose=True)
best_params = results['top_params']
loss = results['top_score']
total_time = results['total_time']
```

### GeneticSearchCV Example:

You can use the `GeneticSearchCV` class as a drop-in replacement for Scikit-Learn's `GridSearchCV`. This 
allows for faster and more complete optimization of your hyperparameters when using Scikit-Learn estimators
and/or pipelines.

```python
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
print("Best parameter (CV score=%0.3f):" % search.best_score_)
print(search.best_params_)
```

### PyPi Project
https://pypi.org/project/geneticpy/

### Contact

Please feel free to email me at brandonschabell@gmail.com with any questions or feedback.
