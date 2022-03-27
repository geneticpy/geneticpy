from asyncio import sleep

import pytest

from geneticpy import optimize
from geneticpy.distributions import *


def test_optimize_simple():
    def fn(params):
        loss = params['x'] + params['y']
        return loss

    param_space = {'x': UniformDistribution(0, 1),
                   'y': UniformDistribution(0, 1000000, 1000)}

    results = optimize(fn, param_space, size=200, generation_count=500, verbose=False, seed=0)
    best_params = results['top_params']
    score = results['top_score']
    time = results['total_time']
    keys = list(best_params.keys())
    keys.sort()
    assert ['x', 'y'] == keys
    assert best_params['x'] < 0.01
    assert best_params['y'] == 0
    assert score < 0.01
    assert 0 < time < 10


def test_optimize_complicated():
    def fn(params):
        loss_x = params['x']
        loss_y = 1 if 18 < params['y'] < 20 else 1000001
        loss_xq = params['xq']
        loss_c = params['c']
        loss = loss_x + loss_y + loss_xq + loss_c
        return loss

    param_space = {'x': UniformDistribution(0, 1),
                   'y': GaussianDistribution(100, 50, low=0),
                   'xq': UniformDistribution(0, 100, q=5),
                   'c': ChoiceDistribution([1000, 3000, 5000], [0.1, 0.7, 0.2])}

    results = optimize(fn, param_space, size=200, generation_count=500, verbose=False, seed=0)
    best_params = results['top_params']
    time = results['total_time']
    keys = list(best_params.keys())
    keys.sort()
    assert ['c', 'x', 'xq', 'y'] == keys
    assert best_params['x'] < 0.01
    assert 18 < best_params['y'] < 20
    assert 1000 == best_params['c']
    assert 0 < time < 10


def test_verbose_mode(capsys):
    def fn(params):
        loss = params['x'] + params['y']
        return loss

    param_space = {'x': UniformDistribution(0, 1),
                   'y': UniformDistribution(0, 1000000, 1000)}

    optimize(fn, param_space, size=200, generation_count=500, verbose=True, seed=0)
    out, err = capsys.readouterr()
    assert 'Optimizing parameters: ' in err


def test_verbose_mode_false(capsys):
    def fn(params):
        loss = params['x'] + params['y']
        return loss

    param_space = {'x': UniformDistribution(0, 1),
                   'y': UniformDistribution(0, 1000000, 1000)}

    optimize(fn, param_space, size=200, generation_count=500, verbose=False, seed=0)
    out, err = capsys.readouterr()
    assert 'Optimizing parameters: ' not in err


def test_constant_parameter():
    def fn(params):
        loss = params['x'] + params['y'] + params['z']
        return loss

    param_space = {'x': UniformDistribution(0, 1),
                   'y': UniformDistribution(0, 1000000, 1000),
                   'z': -50,
                   'zz': 'test',
                   'zzz': {},
                   'zzzz': None,
                   'zzzzz': [1, 2, None, {}, [1, 2]]}

    results = optimize(fn, param_space, size=200, generation_count=500, verbose=False, seed=0)
    best_params = results['top_params']
    score = results['top_score']
    time = results['total_time']
    keys = list(best_params.keys())
    keys.sort()
    assert ['x', 'y', 'z', 'zz', 'zzz', 'zzzz', 'zzzzz'] == keys
    assert best_params['x'] < 0.01
    assert best_params['y'] == 0
    assert best_params['z'] == -50
    assert best_params['zz'] == 'test'
    assert best_params['zzz'] == {}
    assert best_params['zzzz'] is None
    assert best_params['zzzzz'] == [1, 2, None, {}, [1, 2]]
    assert score < -49
    assert 0 < time < 10


def test_target_loss_minimize():
    def fn(params):
        loss = params['x'] + params['y']
        return loss

    param_space = {'x': UniformDistribution(0, 5, q=1),
                   'y': UniformDistribution(0, 1)}

    results = optimize(fn=fn,
                       param_space=param_space,
                       size=200,
                       generation_count=50000,
                       verbose=False,
                       target=1,
                       seed=0)
    score = results['top_score']
    time = results['total_time']

    assert score <= 1
    assert time < 0.1


def test_target_loss_maximize():
    def fn(params):
        loss = params['x'] + params['y']
        return loss

    param_space = {'x': UniformDistribution(0, 5, q=1),
                   'y': UniformDistribution(0, 1)}

    results = optimize(fn=fn,
                       param_space=param_space,
                       size=200,
                       generation_count=50000,
                       maximize_fn=True,
                       verbose=False,
                       target=5,
                       seed=0)
    score = results['top_score']
    time = results['total_time']

    assert score >= 5
    assert time < 0.1


def test_random_seed():
    def fn(params):
        loss = params['x']
        return loss

    param_space = {'x': UniformDistribution(0, 1000000)}
    results1 = optimize(fn=fn, param_space=param_space, size=200, generation_count=500, verbose=False, seed=123)
    best_params1 = results1['top_params']
    score1 = results1['top_score']

    results2 = optimize(fn=fn, param_space=param_space, size=200, generation_count=500, verbose=False, seed=123)
    best_params2 = results2['top_params']
    score2 = results2['top_score']

    results3 = optimize(fn=fn, param_space=param_space, size=200, generation_count=500, verbose=False, seed=124)
    best_params3 = results3['top_params']

    assert best_params1 == best_params2
    assert score1 == score2
    assert best_params1 != best_params3
    assert best_params1 != best_params3


def test_loss_function_none():
    def fn(params):
        return None

    param_space = {'x': UniformDistribution(0, 5)}

    with pytest.raises(Exception):
        optimize(fn=fn, param_space=param_space, size=200, generation_count=50000)


def test_optimize_tqdm_count(capsys):
    def fn(params):
        loss = params['x'] + params['y']
        return loss


    param_space = {'x': UniformDistribution(0, 1),
                   'y': UniformDistribution(0, 1)}

    optimize(fn, param_space, size=50, generation_count=15, verbose=True)
    out, err = capsys.readouterr()
    assert '425/425' in err


def test_async_score():
    async def fn_async(params):
        loss = params['x'] + params['y']
        await sleep(1)
        return loss

    param_space = {'x': UniformDistribution(0, 1),
                   'y': UniformDistribution(0, 1, q=1)}

    response = optimize(fn_async, param_space, size=50, generation_count=3, verbose=True, seed=0)
    best_params = response['top_params']
    score = response['top_score']
    time = response['total_time']
    keys = list(best_params.keys())
    keys.sort()
    assert ['x', 'y'] == keys
    assert best_params['x'] < 0.1
    assert best_params['y'] == 0
    assert score < 0.1
    assert 3 < time < 10
