# Use uv for all project commands
UV := uv run

# --- Formatting --------------------------------------------------------------

.PHONY: format
format:  ## Format code using Ruff
	$(UV) ruff format

# --- Linting -----------------------------------------------------------------

.PHONY: lint
lint:  ## Run Ruff linter and autofix issues
	$(UV) ruff check --fix

# --- Type checking -----------------------------------------------------------

.PHONY: typecheck
typecheck:  ## Run mypy type checking
	$(UV) mypy

# --- Testing -----------------------------------------------------------------

.PHONY: test
test:  ## Run pytest quietly
	$(UV) pytest -q

.PHONY: test-verbose
test-verbose:  ## Run pytest with verbose output
	$(UV) pytest -v

.PHONY: coverage
coverage:  ## Run tests with coverage report
	$(UV) coverage run -m pytest tests/
	$(UV) coverage report

# --- Building ----------------------------------------------------------------

.PHONY: build
build:  ## Build distribution packages
	uv build

# --- Combined tasks ----------------------------------------------------------

.PHONY: check
check: format lint typecheck test  ## Run all validation checks

# --- Utility -----------------------------------------------------------------

.PHONY: clean
clean:  ## Remove build artifacts and cache files
	rm -rf build/ dist/ *.egg-info .pytest_cache .mypy_cache .ruff_cache
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name '*.pyc' -delete
	find . -type f -name '*.pyo' -delete
	find . -type f -name '.coverage' -delete

.PHONY: help
help:  ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-18s\033[0m %s\n", $$1, $$2}'
