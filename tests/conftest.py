"""Pytest configuration and shared fixtures for jactus tests.

This module provides common fixtures and configuration for all tests.
"""

from datetime import datetime, timezone
from typing import Any

import jax
import pytest


@pytest.fixture
def sample_dates() -> dict[str, datetime]:
    """Provide common date fixtures for testing.

    Returns:
        Dictionary of commonly used dates in tests
    """
    return {
        "today": datetime(2024, 1, 15, tzinfo=timezone.utc),
        "past": datetime(2023, 1, 1, tzinfo=timezone.utc),
        "future": datetime(2025, 12, 31, tzinfo=timezone.utc),
        "maturity": datetime(2030, 12, 31, tzinfo=timezone.utc),
        "status_date": datetime(2024, 1, 1, tzinfo=timezone.utc),
        "initial_exchange": datetime(2024, 1, 1, tzinfo=timezone.utc),
    }


@pytest.fixture
def jax_random_key() -> jax.Array:
    """Provide a JAX random number generator key.

    Returns:
        JAX PRNG key for random number generation
    """
    return jax.random.PRNGKey(42)


@pytest.fixture
def tolerance() -> dict[str, float]:
    """Provide numerical tolerance values for float comparisons.

    Returns:
        Dictionary with different tolerance levels
    """
    return {
        "rtol": 1e-5,  # Relative tolerance
        "atol": 1e-8,  # Absolute tolerance
        "strict_rtol": 1e-10,  # Strict relative tolerance
        "strict_atol": 1e-12,  # Strict absolute tolerance
    }


@pytest.fixture
def mock_risk_factor_observer() -> Any:
    """Provide a mock risk factor observer for testing.

    Returns:
        Mock observer that returns fixed values without external data
    """

    class MockRiskFactorObserver:
        """Simple mock observer for testing."""

        def __init__(self) -> None:
            """Initialize with default values."""
            self.fixed_rate = 0.05
            self.fixed_fx_rate = 1.1

        def get_rate(self, time: datetime, market_id: str) -> float:
            """Return a fixed interest rate."""
            return self.fixed_rate

        def get_fx_rate(self, time: datetime, currency_pair: str) -> float:
            """Return a fixed exchange rate."""
            return self.fixed_fx_rate

    return MockRiskFactorObserver()


@pytest.fixture(autouse=True)
def reset_jax_config() -> None:
    """Reset JAX configuration before each test.

    This ensures tests don't interfere with each other's JAX settings.
    """
    # Reset JAX default backend and precision settings
    # This is autouse so it runs before every test
    yield
    # Cleanup after test if needed
    jax.clear_caches()


@pytest.fixture
def sample_contract_attributes() -> dict[str, Any]:
    """Provide sample contract attributes for testing.

    Returns:
        Dictionary with common contract attributes
    """
    return {
        "contract_id": "TEST-001",
        "contract_type": "PAM",  # Principal at Maturity
        "status_date": datetime(2024, 1, 1, tzinfo=timezone.utc),
        "contract_role": "RPA",  # Real Position Asset
        "currency": "USD",
        "nominal_value": 10000.0,
        "initial_exchange_date": datetime(2024, 1, 1, tzinfo=timezone.utc),
        "maturity_date": datetime(2029, 1, 1, tzinfo=timezone.utc),
        "nominal_interest_rate": 0.05,
    }


# Configure pytest markers
def pytest_configure(config: Any) -> None:
    """Configure custom pytest markers.

    Args:
        config: Pytest configuration object
    """
    config.addinivalue_line("markers", "slow: mark test as slow running")
    config.addinivalue_line("markers", "integration: mark test as integration test")
    config.addinivalue_line("markers", "unit: mark test as unit test")
    config.addinivalue_line("markers", "requires_gpu: mark test as requiring GPU")
