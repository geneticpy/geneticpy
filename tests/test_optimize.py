import asyncio
import math
from asyncio import sleep

import pytest
from geneticpy import optimize, optimize_async
from geneticpy.distributions import ChoiceDistribution, GaussianDistribution, UniformDistribution
from geneticpy.optimize_function import OptimizeResult


def test_optimize_simple() -> None:
    def fn(params: dict[str, float]) -> float:
        loss = params["x"] + params["y"]
        return loss

    param_space = {"x": UniformDistribution(0, 1), "y": UniformDistribution(0, 1000000, 1000)}

    results = optimize(fn, param_space, size=200, generation_count=500, verbose=False, seed=0)
    best_params = results.best_params
    score = results.best_score
    time = results.total_time
    keys = list(best_params.keys())
    keys.sort()
    assert keys == ["x", "y"]
    assert best_params["x"] < 0.01
    assert best_params["y"] == 0
    assert score < 0.01
    assert 0 < time < 10


def test_optimize_complicated() -> None:
    def fn(params: dict[str, float]) -> float:
        loss_x = params["x"]
        loss_y = 1 if 18 < params["y"] < 20 else 1000001
        loss_xq = params["xq"]
        loss_c = params["c"]
        loss = loss_x + loss_y + loss_xq + loss_c
        return loss

    param_space = {
        "x": UniformDistribution(0, 1),
        "y": GaussianDistribution(100, 50, low=0),
        "xq": UniformDistribution(0, 100, q=5),
        "c": ChoiceDistribution([1000, 3000, 5000], [0.1, 0.7, 0.2]),
    }

    results = optimize(fn, param_space, size=200, generation_count=500, verbose=False, seed=0)
    best_params = results.best_params
    time = results.total_time
    keys = list(best_params.keys())
    keys.sort()
    assert keys == ["c", "x", "xq", "y"]
    assert best_params["x"] < 0.01
    assert 18 < best_params["y"] < 20
    assert best_params["c"] == 1000
    assert 0 < time < 10


def test_verbose_mode(capsys: pytest.CaptureFixture) -> None:
    def fn(params: dict[str, float]) -> float:
        loss = params["x"] + params["y"]
        return loss

    param_space = {"x": UniformDistribution(0, 1), "y": UniformDistribution(0, 1000000, 1000)}

    optimize(fn, param_space, size=200, generation_count=500, verbose=True, seed=0)
    _, err = capsys.readouterr()
    assert "Optimizing parameters: " in err


def test_verbose_mode_false(capsys: pytest.CaptureFixture) -> None:
    def fn(params: dict[str, float]) -> float:
        loss = params["x"] + params["y"]
        return loss

    param_space = {"x": UniformDistribution(0, 1), "y": UniformDistribution(0, 1000000, 1000)}

    optimize(fn, param_space, size=200, generation_count=500, verbose=False, seed=0)
    _, err = capsys.readouterr()
    assert "Optimizing parameters: " not in err


def test_constant_parameter() -> None:
    def fn(params: dict[str, float]) -> float:
        loss = params["x"] + params["y"] + params["z"]
        return loss

    param_space = {
        "x": UniformDistribution(0, 1),
        "y": UniformDistribution(0, 1000000, 1000),
        "z": -50,
        "zz": "test",
        "zzz": {},
        "zzzz": None,
        "zzzzz": [1, 2, None, {}, [1, 2]],
    }

    results = optimize(fn, param_space, size=200, generation_count=500, verbose=False, seed=0)
    best_params = results.best_params
    score = results.best_score
    time = results.total_time
    keys = list(best_params.keys())
    keys.sort()
    assert keys == ["x", "y", "z", "zz", "zzz", "zzzz", "zzzzz"]
    assert best_params["x"] < 0.01
    assert best_params["y"] == 0
    assert best_params["z"] == -50
    assert best_params["zz"] == "test"
    assert best_params["zzz"] == {}
    assert best_params["zzzz"] is None
    assert best_params["zzzzz"] == [1, 2, None, {}, [1, 2]]
    assert score < -49
    assert 0 < time < 10


def test_target_loss_minimize() -> None:
    def fn(params: dict[str, float]) -> float:
        loss = params["x"] + params["y"]
        return loss

    param_space = {"x": UniformDistribution(0, 5, q=1), "y": UniformDistribution(0, 1)}

    results = optimize(
        fn=fn, param_space=param_space, size=200, generation_count=50000, verbose=False, target=1, seed=0
    )
    score = results.best_score
    time = results.total_time

    assert score <= 1
    assert time < 0.1


def test_target_loss_maximize() -> None:
    def fn(params: dict[str, float]) -> float:
        loss = params["x"] + params["y"]
        return loss

    param_space = {"x": UniformDistribution(0, 5, q=1), "y": UniformDistribution(0, 1)}

    results = optimize(
        fn=fn,
        param_space=param_space,
        size=200,
        generation_count=50000,
        maximize_fn=True,
        verbose=False,
        target=5,
        seed=0,
    )
    score = results.best_score
    time = results.total_time

    assert score >= 5
    assert time < 0.1


def test_random_seed() -> None:
    def fn(params: dict[str, float]) -> float:
        loss = params["x"]
        return loss

    param_space = {"x": UniformDistribution(0, 1000000)}
    results1 = optimize(fn=fn, param_space=param_space, size=200, generation_count=500, verbose=False, seed=123)
    best_params1 = results1.best_params
    score1 = results1.best_score

    results2 = optimize(fn=fn, param_space=param_space, size=200, generation_count=500, verbose=False, seed=123)
    best_params2 = results2.best_params
    score2 = results2.best_score

    results3 = optimize(fn=fn, param_space=param_space, size=200, generation_count=500, verbose=False, seed=124)
    best_params3 = results3.best_params

    assert best_params1 == best_params2
    assert score1 == score2
    assert best_params1 != best_params3
    assert best_params1 != best_params3


def test_loss_function_none() -> None:
    def fn(params: dict[str, float]) -> float | None:
        return None

    param_space = {"x": UniformDistribution(0, 5)}

    with pytest.raises(ValueError):
        optimize(fn=fn, param_space=param_space, size=200, generation_count=50000)  # type: ignore[arg-type]


def test_optimize_tqdm_count(capsys: pytest.CaptureFixture) -> None:
    def fn(params: dict[str, float]) -> float:
        loss = params["x"] + params["y"]
        return loss

    param_space = {"x": UniformDistribution(0, 1), "y": UniformDistribution(0, 1)}

    optimize(fn, param_space, size=50, generation_count=15, verbose=True)
    _, err = capsys.readouterr()
    assert "425/425" in err


def test_async_score() -> None:
    async def fn_async(params: dict[str, float]) -> float:
        loss = params["x"] + params["y"]
        await sleep(1)
        return loss

    param_space = {"x": UniformDistribution(0, 1), "y": UniformDistribution(0, 1, q=1)}

    response = optimize(fn_async, param_space, size=50, generation_count=3, verbose=True, seed=0)
    best_params = response.best_params
    score = response.best_score
    time = response.total_time
    keys = list(best_params.keys())
    keys.sort()
    assert keys == ["x", "y"]
    assert best_params["x"] < 0.1
    assert best_params["y"] == 0
    assert score < 0.1
    assert 3 < time < 10


def test_target_reached_on_equality() -> None:
    calls = 0

    def fn(params: dict[str, float]) -> float:
        nonlocal calls
        calls += 1
        return 0.0

    param_space = {"x": UniformDistribution(0, 1)}

    optimize(fn=fn, param_space=param_space, size=10, generation_count=5, target=0.0, seed=0)

    assert calls == 10


@pytest.mark.parametrize("maximize_fn", [True, False])
def test_nan_never_reported_as_best(maximize_fn: bool) -> None:
    def fn(params: dict[str, float]) -> float:
        return float("nan") if params["x"] > 0.5 else params["x"]

    for seed in range(10):
        result = optimize(
            fn, {"x": UniformDistribution(0, 1)}, size=20, generation_count=5, seed=seed, maximize_fn=maximize_fn
        )
        assert not math.isnan(result.best_score)


def test_constants_only_param_space() -> None:
    result = optimize(lambda params: params["x"], {"x": 5}, size=10, generation_count=3, seed=0)
    assert result.best_params == {"x": 5}


def test_single_event_loop_per_optimize() -> None:
    loops: set[asyncio.AbstractEventLoop] = set()

    async def fn(params: dict[str, float]) -> float:
        loops.add(asyncio.get_running_loop())
        return params["x"]

    optimize(fn, {"x": UniformDistribution(0, 1)}, size=10, generation_count=5, seed=0)
    assert len(loops) == 1


def test_callable_returning_awaitable() -> None:
    async def score(params: dict[str, float]) -> float:
        return params["x"]

    result = optimize(lambda params: score(params), {"x": UniformDistribution(0, 1)}, size=10, generation_count=2)
    assert 0 <= result.best_score <= 1


def test_optimize_async_in_running_loop() -> None:
    async def main() -> OptimizeResult:
        return await optimize_async(lambda params: params["x"], {"x": UniformDistribution(0, 1)}, size=10, seed=0)

    result = asyncio.run(main())
    assert 0 <= result.best_score <= 1


def test_optimize_in_running_loop_points_to_optimize_async() -> None:
    async def main() -> None:
        optimize(lambda params: params["x"], {"x": UniformDistribution(0, 1)}, size=10)

    with pytest.raises(RuntimeError, match="optimize_async"):
        asyncio.run(main())
