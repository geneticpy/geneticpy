# GeneticPy

[![Build Status](https://travis-ci.com/geneticpy/geneticpy.svg?branch=master)](https://travis-ci.com/geneticpy/geneticpy)
[![PyPI version](https://badge.fury.io/py/geneticpy.svg)](https://badge.fury.io/py/geneticpy)

GeneticPy is an optimizer that uses a genetic algorithm to quickly search through custom parameter spaces for optimal solutions.

### Installation

GeneticPy requires Python 3.4+

```sh
pip install geneticpy
```

### Example Usage:

A brief example to get you started is included below:

```python
def loss_function(params):
  if params['type'] == 'add':
    return params['x'] + params['y']
  elif params['type'] == 'multiply':
    return params['x'] * params['y']

param_space = {'type': geneticpy.ChoiceDistribution(choice_list=['add', 'multiply']),
               'x': geneticpy.UniformDistribution(low=5, high=10, q=1),
               'y': geneticpy.GaussianDistribution(mean=0, standard_deviation=1)}

best_params, loss = geneticpy.optimize(loss_function, param_space, size=200, generation_count=500, verbose=False)
```

### PyPi Project
https://pypi.org/project/geneticpy/

### Contact

Please feel free to email me at brandonschabell@gmail.com with any questions or feedback.
