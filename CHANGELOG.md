# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Python 3.15 support
- `optimize_async` for running an optimization inside an existing event loop (e.g. Jupyter notebooks or async applications)
- `py.typed` marker so type checkers use GeneticPy's type hints

### Changed
- `GaussianDistribution`, `ExponentialDistribution`, and `LogNormalDistribution` now sample from the truncated distribution when `low`/`high` are set, instead of clipping out-of-range samples to the bound. Breeding samples between the parents' values instead of mostly copying one parent. Results for a given `seed` will differ from previous versions.
- Objective functions that return an awaitable (e.g. a lambda wrapping an async function) are now awaited

### Removed
- Unused `pandas` dev dependency

### Fixed
- Breeding of `GaussianDistribution`, `ExponentialDistribution`, and `LogNormalDistribution` parameters collapsed to a single parent's value when the first parent's value was larger than the second's
- `ChoiceDistribution` returned numpy types instead of the original objects, stringified mixed-type choice lists (e.g. `True` became `'True'`), and raised on lists of tuples
- `adaptive_mutation` had no effect unless `patience` was also set, and did not compound across generations
- `target` now stops optimization when a score equal to the target is reached, as documented
- Mutated individuals could be mutated again within the same generation; `mutate_chance=1.0` caused an infinite loop
- Diversity injection could replace retained individuals when `retain_percentage` was above 0.9
- NaN scores could be reported as the best result, most often when `maximize_fn=True`
- Quantized (`q`) values could fall outside the distribution's `low`/`high` bounds
- `optimize` created a new event loop every generation, breaking async objective functions that hold loop-bound resources. Calling it from a running event loop now raises an error pointing to `optimize_async`
- Parameter spaces containing only constants raised `IndexError`
- `ChoiceDistribution` ignored tuple probabilities and raised on numpy array probabilities
- `DeprecationWarning` from `asyncio.iscoroutinefunction`, which is removed in Python 3.16

## [2.0.0] - 2025-10-11

### Changed
- **BREAKING**: Removed scikit-learn dependency and `GeneticSearchCV` class
- **BREAKING**: Minimum Python version increased from 3.8 to 3.10
- Migrated from setuptools to hatchling build backend
- Migrated from pip to uv for dependency management
- Modernized CI/CD workflows to use uv
- Updated README with modern development workflow

### Added
- Comprehensive type hints across entire codebase
- NumPy-style docstrings on all public APIs
- Makefile with helpful development targets (test, lint, format, typecheck, build, coverage, clean)
- MyPy type checking with strict mode enabled
- Ruff for linting and formatting with docstring validation
- Coverage reporting configuration
- Python 3.13 and 3.14 support
- **Tournament selection** option for improved parent selection pressure (`use_tournament_selection`, `tournament_size` parameters)
- **Diversity protection** with automatic population diversity monitoring and injection when diversity falls below threshold
  - Handles both numeric parameters (variance-based) and categorical choice parameters (unique ratio)
  - Automatically injects random individuals when diversity falls below threshold
- **Adaptive mutation** that automatically adjusts exploration vs exploitation based on optimization progress
- **Early stopping** via `patience` parameter to halt optimization when no improvement is seen for N generations
- **Enhanced mutation** with `mutation_rate` parameter (0.0-1.0) for probabilistic per-parameter mutation
- Comprehensive test suite with 15 new tests covering all new features (64 total tests, 97% coverage)

### Fixed
- ReadTheDocs configuration for proper documentation building
- Docstring formatting for Sphinx compatibility
- Infinite loop bug in breeding logic where `if set1 != set2` condition could hang indefinitely
- Incomplete random seeding - now seeds both `random` and `numpy.random` for full reproducibility
- Weak elitism protection - top performers are now preserved correctly across generations
- Improved breeding algorithm to always produce valid offspring

## [1.4.0] - Previous Release

### Added
- Python 3.11 support
- GitHub Actions workflow updates

### Fixed
- Documentation improvements

