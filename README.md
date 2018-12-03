# geneticpy
Hyperparameter optimization based on a genetic algorithm.

Stable Release: N/A

Beta Release: v0.0.1

[![Build Status](https://travis-ci.com/geneticpy/geneticpy.svg?branch=master)](https://travis-ci.com/geneticpy/geneticpy)

Install Using:
```
pip install geneticpy
```

Example Usage:

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
