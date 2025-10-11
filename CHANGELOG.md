# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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

