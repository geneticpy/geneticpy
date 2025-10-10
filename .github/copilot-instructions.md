# GeneticPy Copilot Instructions

## Architecture Overview

GeneticPy is a lightweight genetic algorithm optimizer focused on general parameter optimization through the `optimize()` function.

### Core Components

**Population-based Evolution**: The `Population` class manages genetic algorithm mechanics:
- Maintains parameter sets that evolve through breeding and mutation
- Supports both sync and async objective functions via automatic wrapping
- Uses `ParameterSet` instances that can mutate individual parameters or breed with others

**Distribution System**: All parameter spaces use `DistributionBase` subclasses:
- `UniformDistribution`, `GaussianDistribution`, `ChoiceDistribution`, etc. in `distributions/__init__.py`
- Each implements `pull_value()` for random sampling and `pull_constrained_value()` for breeding
- Support quantization (`q` parameter) and bounds constraints

**Simple API**: 
- `optimize_function.py`: Functional API for general optimization
- No external ML library dependencies - pure genetic algorithm implementation

## Development Patterns

**Parameter Space Definition**: Always use distribution objects, not raw values:
```python
# Correct
param_space = {'x': UniformDistribution(0, 1, q=0.1)}
# Wrong  
param_space = {'x': [0, 0.1, 0.2, 0.3]}
```

**Async Support**: The population automatically wraps sync functions in async. Both work:
```python
def sync_fn(params): return params['x'] + params['y']
async def async_fn(params): return params['x'] + params['y']
```

**Testing Strategy**: 
- Use small populations (`size=10`) and few generations (`generation_count=2`) for fast tests
- Always set `seed=0` for reproducible results in tests
- Focus on simple optimization functions and parameter space validation

## Key File Relationships

- `__init__.py`: Exports main APIs and all distribution classes
- `parameter_set.py`: Individual solutions that can mutate/breed
- `population.py`: Manages evolution of parameter sets
- `optimize_function.py`: Simple functional interface

## Testing & Build

**Run tests**: `pytest tests/` (uses matrix testing across Python 3.8-3.13)

**Dependencies**: Minimal core deps (numpy, tqdm) with optional test/docs extras

**Packaging**: Uses modern `pyproject.toml` (PEP 621) with explicit version in `__init__.py`