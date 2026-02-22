"""Testing utilities and helper functions for jactus tests."""

from datetime import datetime, timedelta, timezone
from typing import Any

import jax.numpy as jnp
import numpy as np


def assert_dates_equal(
    date1: datetime,
    date2: datetime,
    tolerance: timedelta = timedelta(seconds=1),
    msg: str | None = None,
) -> None:
    """Compare dates with timezone awareness and tolerance.

    Args:
        date1: First date to compare
        date2: Second date to compare
        tolerance: Maximum acceptable difference between dates
        msg: Optional custom error message

    Raises:
        AssertionError: If dates differ by more than tolerance
    """
    diff = abs(date1 - date2)
    if diff > tolerance:
        error_msg = msg or f"Dates differ by {diff}: {date1} vs {date2}"
        raise AssertionError(error_msg)


def assert_arrays_close(
    actual: jnp.ndarray,
    expected: jnp.ndarray,
    rtol: float = 1e-5,
    atol: float = 1e-8,
    msg: str | None = None,
) -> None:
    """JAX array comparison with tolerance.

    Args:
        actual: Actual array values
        expected: Expected array values
        rtol: Relative tolerance
        atol: Absolute tolerance
        msg: Optional custom error message

    Raises:
        AssertionError: If arrays are not close within tolerances
    """
    if not jnp.allclose(actual, expected, rtol=rtol, atol=atol):
        max_diff = jnp.max(jnp.abs(actual - expected))
        error_msg = msg or (
            f"Arrays not close within rtol={rtol}, atol={atol}. "
            f"Max difference: {max_diff}\n"
            f"Actual: {actual}\n"
            f"Expected: {expected}"
        )
        raise AssertionError(error_msg)


def assert_cashflows_valid(cashflows: dict[str, Any], msg: str | None = None) -> None:
    """Validate cashflow structure and values.

    Args:
        cashflows: Dictionary containing cashflow data with keys:
                  - 'times': array of payment times
                  - 'amounts': array of payment amounts
                  - 'types': array of event types (optional)
        msg: Optional custom error message

    Raises:
        AssertionError: If cashflow structure is invalid
    """
    # Check required keys
    if "times" not in cashflows or "amounts" not in cashflows:
        error_msg = msg or "Cashflows must contain 'times' and 'amounts' keys"
        raise AssertionError(error_msg)

    times = cashflows["times"]
    amounts = cashflows["amounts"]

    # Check shapes match
    if len(times) != len(amounts):
        error_msg = msg or (
            f"Length mismatch: times has {len(times)} elements, amounts has {len(amounts)} elements"
        )
        raise AssertionError(error_msg)

    # Check for NaN or Inf values
    if jnp.any(jnp.isnan(amounts)) or jnp.any(jnp.isinf(amounts)):
        error_msg = msg or "Cashflow amounts contain NaN or Inf values"
        raise AssertionError(error_msg)

    # Check times are sorted
    if not jnp.all(times[:-1] <= times[1:]):
        error_msg = msg or "Cashflow times are not sorted in ascending order"
        raise AssertionError(error_msg)


def create_test_contract(
    contract_type: str = "PAM",
    contract_id: str = "TEST-001",
    nominal: float = 10000.0,
    rate: float = 0.05,
    start_date: datetime | None = None,
    maturity_date: datetime | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Factory function for creating test contracts.

    Args:
        contract_type: ACTUS contract type (e.g., 'PAM', 'ANN', 'NAM')
        contract_id: Unique contract identifier
        nominal: Nominal principal amount
        rate: Nominal interest rate
        start_date: Contract start date
        maturity_date: Contract maturity date
        **kwargs: Additional contract-specific attributes

    Returns:
        Dictionary with contract attributes suitable for testing
    """
    if start_date is None:
        start_date = datetime(2024, 1, 1, tzinfo=timezone.utc)
    if maturity_date is None:
        maturity_date = datetime(2029, 1, 1, tzinfo=timezone.utc)

    contract: dict[str, Any] = {
        "contract_id": contract_id,
        "contract_type": contract_type,
        "status_date": start_date,
        "contract_role": "RPA",
        "currency": "USD",
        "nominal_value": nominal,
        "initial_exchange_date": start_date,
        "maturity_date": maturity_date,
        "nominal_interest_rate": rate,
    }

    # Update with any additional kwargs
    contract.update(kwargs)

    return contract


def create_mock_market_data(
    base_rate: float = 0.05, num_points: int = 100, volatility: float = 0.01
) -> dict[str, np.ndarray]:
    """Create mock market data for testing.

    Args:
        base_rate: Base interest rate level
        num_points: Number of data points to generate
        volatility: Standard deviation of random fluctuations

    Returns:
        Dictionary containing mock market data:
        - 'times': Array of time points
        - 'rates': Array of interest rates
        - 'fx_rates': Array of exchange rates
    """
    times = np.linspace(0, 10, num_points)  # 10 years of data
    rates = base_rate + volatility * np.random.randn(num_points)
    fx_rates = 1.0 + 0.1 * np.sin(times) + 0.05 * np.random.randn(num_points)

    return {
        "times": times,
        "rates": np.maximum(rates, 0.001),  # Keep rates positive
        "fx_rates": np.maximum(fx_rates, 0.5),  # Keep FX rates reasonable
    }


def assert_no_exceptions_raised(func: Any, *args: Any, **kwargs: Any) -> Any:
    """Assert that a function call does not raise any exceptions.

    Args:
        func: Function to call
        *args: Positional arguments to pass to func
        **kwargs: Keyword arguments to pass to func

    Returns:
        Result of func(*args, **kwargs)

    Raises:
        AssertionError: If func raises any exception
    """
    try:
        return func(*args, **kwargs)
    except Exception as e:
        raise AssertionError(f"Function {func.__name__} raised {type(e).__name__}: {e}") from e


def get_sample_schedule(
    start_date: datetime,
    end_date: datetime,
    frequency_months: int = 3,
) -> list[datetime]:
    """Generate a sample payment schedule for testing.

    Args:
        start_date: Schedule start date
        end_date: Schedule end date
        frequency_months: Payment frequency in months

    Returns:
        List of payment dates
    """
    schedule = []
    current_date = start_date

    while current_date <= end_date:
        schedule.append(current_date)
        # Simple month addition (not production quality)
        year = current_date.year
        month = current_date.month + frequency_months
        while month > 12:
            month -= 12
            year += 1
        try:
            current_date = current_date.replace(year=year, month=month)
        except ValueError:
            # Handle end of month issues
            current_date = current_date.replace(year=year, month=month, day=1)

    return schedule
