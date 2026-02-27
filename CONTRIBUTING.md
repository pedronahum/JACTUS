# Contributing to JACTUS

Thank you for your interest in contributing to JACTUS! This document provides guidelines and instructions for contributing.

## Code of Conduct

By participating in this project, you agree to abide by our code of conduct:

- Be respectful and inclusive
- Welcome newcomers and help them get started
- Focus on constructive criticism
- Accept feedback graciously
- Prioritize the community's well-being

## Getting Started

### Development Setup

1. **Fork and clone the repository**

```bash
git clone https://github.com/pedronahum/JACTUS.git
cd JACTUS
```

2. **Set up the development environment**

```bash
./scripts/setup_dev.sh
```

Or manually:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -e ".[dev,docs,viz]"
pre-commit install
```

3. **Verify the setup**

```bash
make test
make quality
```

## Development Workflow

### 1. Create a Branch

Create a feature branch from `main`:

```bash
git checkout -b feature/your-feature-name
```

Branch naming conventions:
- `feature/` - New features
- `fix/` - Bug fixes
- `docs/` - Documentation updates
- `refactor/` - Code refactoring
- `test/` - Test additions or updates

### 2. Make Your Changes

- Write clear, concise code following our style guide
- Add tests for new functionality
- Update documentation as needed
- Keep commits focused and atomic

### 3. Test Your Changes

```bash
# Run all tests
make test

# Run with coverage
make test-cov

# Run specific test types
pytest -m unit
pytest -m integration
```

### 4. Code Quality Checks

Before committing, ensure your code passes all quality checks:

```bash
# Format code
make format

# Run linter
make lint

# Type checking
make typecheck

# All checks
make quality
```

### 5. Commit Your Changes

Write clear commit messages following conventional commits:

```bash
git commit -m "feat: add support for NAM contract type"
git commit -m "fix: correct day count calculation in ACT/365"
git commit -m "docs: update API reference for observers"
```

Commit message format:
- `feat:` - New feature
- `fix:` - Bug fix
- `docs:` - Documentation changes
- `test:` - Test updates
- `refactor:` - Code refactoring
- `perf:` - Performance improvements
- `chore:` - Maintenance tasks

### 6. Push and Create Pull Request

```bash
git push origin feature/your-feature-name
```

Then create a Pull Request on GitHub.

## Code Style Guidelines

### Python Style

We follow [PEP 8](https://pep8.org/) with some modifications:

- **Line length**: 100 characters (enforced by Black)
- **Formatting**: Use Black for code formatting
- **Imports**: Sorted with isort (integrated in Ruff)
- **Type hints**: Required for all public APIs
- **Docstrings**: Google-style docstrings

### Example

```python
"""Module docstring describing the module's purpose."""

from typing import Optional

import jax.numpy as jnp
from jax import jit

from jactus.exceptions import ContractValidationError


def calculate_cashflow(
    nominal: float,
    rate: float,
    time_fraction: float,
    contract_id: Optional[str] = None,
) -> float:
    """Calculate interest cashflow for a period.

    Args:
        nominal: Nominal principal amount
        rate: Interest rate (annualized)
        time_fraction: Time fraction for the period
        contract_id: Optional contract identifier for error reporting

    Returns:
        Cashflow amount

    Raises:
        ContractValidationError: If inputs are invalid

    Example:
        >>> calculate_cashflow(10000, 0.05, 0.25)
        125.0
    """
    if nominal < 0:
        raise ContractValidationError(
            "Nominal must be non-negative",
            context={"contract_id": contract_id, "nominal": nominal}
        )

    return nominal * rate * time_fraction
```

## Testing Requirements

### Test Coverage

- Maintain minimum 90% code coverage for new code
- Write both unit tests and integration tests
- Use property-based testing (Hypothesis) where appropriate

### Behavioral Observer Tests

Behavioral risk factor observers (`PrepaymentSurfaceObserver`, `DepositTransactionObserver`) and their supporting utilities (`Surface2D`, `LabeledSurface2D`) have dedicated test files:

- `tests/unit/observers/test_behavioral.py` — tests for behavioral observer implementations
- `tests/unit/utilities/test_surface.py` — tests for 2D surface interpolation utilities

When modifying or extending the behavioral observer framework, ensure these tests pass and add coverage for any new behavior.

### Test Structure

```python
"""Test module for cashflow calculations."""

import pytest
from hypothesis import given, strategies as st

from jactus.functions import calculate_cashflow
from jactus.exceptions import ContractValidationError


class TestCashflowCalculation:
    """Test suite for cashflow calculations."""

    def test_basic_cashflow(self):
        """Test basic interest cashflow calculation."""
        result = calculate_cashflow(10000, 0.05, 0.25)
        assert result == pytest.approx(125.0)

    def test_negative_nominal_raises_error(self):
        """Test that negative nominal raises validation error."""
        with pytest.raises(ContractValidationError):
            calculate_cashflow(-10000, 0.05, 0.25)

    @given(
        nominal=st.floats(min_value=0, max_value=1e9),
        rate=st.floats(min_value=0, max_value=1),
        time_fraction=st.floats(min_value=0, max_value=1),
    )
    def test_cashflow_properties(self, nominal, rate, time_fraction):
        """Property-based test for cashflow calculation."""
        result = calculate_cashflow(nominal, rate, time_fraction)
        assert result >= 0
        assert result <= nominal  # Assuming rate and time_fraction <= 1
```

## Documentation

### Docstring Requirements

All public functions, classes, and modules must have docstrings:

```python
def my_function(arg1: int, arg2: str) -> bool:
    """One-line summary.

    Longer description if needed, explaining the function's behavior,
    algorithm, or important details.

    Args:
        arg1: Description of arg1
        arg2: Description of arg2

    Returns:
        Description of return value

    Raises:
        ValueError: When input is invalid
        TypeError: When type is incorrect

    Example:
        >>> my_function(42, "test")
        True

    Note:
        Any important notes or caveats

    See Also:
        related_function: Description
    """
```

### Building Documentation

```bash
# Build docs
make docs

# Serve docs locally
cd docs && make html && python -m http.server --directory _build/html
```

## Pull Request Process

### Before Submitting

- [ ] All tests pass (`make test`)
- [ ] Code quality checks pass (`make quality`)
- [ ] Documentation is updated
- [ ] CHANGELOG.md is updated (if applicable)
- [ ] Commit messages follow conventional commits
- [ ] Branch is up to date with main

### PR Description Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing
Describe the tests you added or ran

## Checklist
- [ ] Tests pass
- [ ] Code quality checks pass
- [ ] Documentation updated
- [ ] CHANGELOG.md updated
```

### Review Process

1. Automated checks must pass (CI/CD)
2. At least one maintainer approval required
3. All review comments must be addressed
4. Branch must be up to date with main

## Issue Reporting

### Bug Reports

Include:
- Description of the bug
- Steps to reproduce
- Expected behavior
- Actual behavior
- Environment (Python version, OS, JAX version)
- Minimal code example

### Feature Requests

Include:
- Clear description of the feature
- Use cases and motivation
- Proposed API or implementation (if any)
- Alternatives considered

## Questions?

- Open a [GitHub Discussion](https://github.com/pedronahum/jactus/discussions)
- Check existing issues and documentation
- Contact maintainer at pnrodriguezh@gmail.com

## License

By contributing, you agree that your contributions will be licensed under the Apache License 2.0.
