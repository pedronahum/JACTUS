.PHONY: help install install-dev test test-cov lint format typecheck docs clean all pre-commit

# Default target
help:
	@echo "Available targets:"
	@echo "  install       - Install package in editable mode"
	@echo "  install-dev   - Install with development dependencies"
	@echo "  test          - Run test suite"
	@echo "  test-cov      - Run tests with coverage"
	@echo "  test-parallel - Run tests in parallel"
	@echo "  lint          - Run ruff linter"
	@echo "  format        - Run black formatter"
	@echo "  typecheck     - Run mypy type checker"
	@echo "  docs          - Build documentation"
	@echo "  docs-serve    - Build and serve documentation"
	@echo "  clean         - Remove build artifacts"
	@echo "  pre-commit    - Install pre-commit hooks"
	@echo "  all           - Run quality checks and tests"

# Installation targets
install:
	pip install -e .

install-dev:
	pip install -e ".[dev,docs,viz]"

# Testing targets
test:
	pytest tests/ -v

test-cov:
	pytest tests/ --cov=jactus --cov-report=html --cov-report=term

test-parallel:
	pytest tests/ -n auto

test-unit:
	pytest tests/unit/ -v -m unit

test-integration:
	pytest tests/integration/ -v -m integration

# Code quality targets
lint:
	ruff check src/ tests/

lint-fix:
	ruff check --fix src/ tests/

format:
	black src/ tests/ examples/

format-check:
	black --check src/ tests/ examples/

typecheck:
	mypy src/jactus

# Pre-commit hooks
pre-commit:
	pre-commit install

pre-commit-run:
	pre-commit run --all-files

# Documentation targets
docs:
	cd docs && make html

docs-serve:
	cd docs && make html && python -m http.server --directory _build/html

docs-clean:
	cd docs && make clean

# Cleanup targets
clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf .ruff_cache/
	rm -rf htmlcov/
	rm -rf .coverage
	rm -rf docs/_build/
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete

# Combined quality checks
quality: format-check lint typecheck
	@echo "All quality checks passed!"

# Run all checks and tests
all: quality test
	@echo "All checks and tests passed!"

# Development setup
setup-dev: install-dev pre-commit
	@echo "Development environment setup complete!"
