"""Financial mathematics utilities for ACTUS contracts.

This module provides mathematical functions for contract calculations including
contract role signs, annuity calculations, and discount factors.

References:
    ACTUS Technical Specification v1.1, Table 1 (Contract Role Signs)
    ACTUS Technical Specification v1.1, Section 5 (Mathematical Functions)
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from jactus.core.time import ActusDateTime
from jactus.core.types import ContractRole, DayCountConvention
from jactus.utilities.conventions import year_fraction


def contract_role_sign(role: ContractRole) -> int:
    """Get the sign (+1 or -1) for a contract role.

    The contract role sign determines the direction of cash flows from the
    perspective of the contract holder.

    Args:
        role: Contract role

    Returns:
        +1 for long/receiving positions, -1 for short/paying positions

    Example:
        >>> contract_role_sign(ContractRole.RPA)  # Receiving party A
        1
        >>> contract_role_sign(ContractRole.RPL)  # Real position lender
        -1

    References:
        ACTUS Technical Specification v1.1, Table 1
    """
    # Use the method from ContractRole enum
    return role.get_sign()


@jax.jit
def contract_role_sign_vectorized(roles: jnp.ndarray) -> jnp.ndarray:
    """Vectorized contract role sign calculation for JAX.

    Args:
        roles: Array of contract role values (as integers)

    Returns:
        Array of signs (+1 or -1)

    Example:
        >>> roles = jnp.array([0, 1, 2])  # RPA, RPL, LG
        >>> signs = contract_role_sign_vectorized(roles)
        >>> signs
        Array([1, -1, 1], dtype=int32)

    Note:
        This function is JIT-compiled for performance.
    """
    # Map role indices to signs according to ACTUS Table 1
    # This matches the ordering in ContractRole enum
    sign_map = jnp.array(
        [
            1,  # RPA - Real Position Asset
            -1,  # RPL - Real Position Liability
            1,  # LG - Long Position
            -1,  # ST - Short Position
            1,  # BUY - Protection Buyer
            -1,  # SEL - Protection Seller
            1,  # RFL - Receive First Leg
            -1,  # PFL - Pay First Leg
            1,  # COL - Collateral Instrument
            1,  # CNO - Close-out Netting Instrument
            -1,  # GUA - Guarantor
            1,  # OBL - Obligee
            1,  # UDL - Underlying
            1,  # UDLP - Underlying Positive
            -1,  # UDLM - Underlying Negative
        ],
        dtype=jnp.int32,
    )

    return sign_map[roles]


def annuity_amount(
    notional: float,
    rate: float,
    tenor: ActusDateTime,  # noqa: ARG001
    maturity: ActusDateTime,  # noqa: ARG001
    n_periods: int,
    day_count_convention: DayCountConvention,  # noqa: ARG001
) -> float:
    """Calculate annuity payment amount.

    Computes the periodic payment for an annuity given notional, rate, and term.
    Uses the formula: A = N * r / (1 - (1 + r)^(-n))

    Args:
        notional: Notional principal amount
        rate: Periodic interest rate (e.g., 0.05 for 5% per period)
        tenor: Start date for year fraction calculation (reserved for future use)
        maturity: End date for year fraction calculation (reserved for future use)
        n_periods: Number of payment periods
        day_count_convention: Day count convention (reserved for future use)

    Returns:
        Annuity payment amount per period

    Example:
        >>> # $100,000 loan at 5% annual rate, 12 monthly payments
        >>> tenor = ActusDateTime(2024, 1, 1, 0, 0, 0)
        >>> maturity = ActusDateTime(2025, 1, 1, 0, 0, 0)
        >>> amount = annuity_amount(100000, 0.05/12, tenor, maturity, 12, DayCountConvention.A360)
        >>> abs(amount - 8560.75) < 1  # Approximately $8,560.75 per month
        True

    References:
        ACTUS Technical Specification v1.1, Section 5.1
    """
    if n_periods == 0:
        return 0.0

    if abs(rate) < 1e-10:
        # For zero or near-zero rates, annuity is just notional / periods
        return notional / n_periods

    # Standard annuity formula: A = N * r / (1 - (1 + r)^(-n))
    denominator = 1.0 - (1.0 + rate) ** (-n_periods)
    return notional * rate / denominator


@jax.jit
def annuity_amount_vectorized(
    notional: jnp.ndarray,
    rate: jnp.ndarray,
    n_periods: jnp.ndarray,
) -> jnp.ndarray:
    """Vectorized annuity calculation for JAX arrays.

    Args:
        notional: Array of notional amounts
        rate: Array of periodic rates
        n_periods: Array of number of periods

    Returns:
        Array of annuity amounts

    Example:
        >>> notionals = jnp.array([100000.0, 200000.0])
        >>> rates = jnp.array([0.05/12, 0.04/12])
        >>> periods = jnp.array([12, 24])
        >>> amounts = annuity_amount_vectorized(notionals, rates, periods)

    Note:
        This function is JIT-compiled for performance.
    """
    # Handle zero periods
    zero_periods = n_periods == 0
    result = jnp.where(zero_periods, 0.0, notional)

    # Handle near-zero rates
    near_zero_rate = jnp.abs(rate) < 1e-10
    simple_annuity = notional / n_periods
    result = jnp.where(near_zero_rate & ~zero_periods, simple_annuity, result)

    # Standard annuity formula
    valid_mask = ~zero_periods & ~near_zero_rate
    denominator = 1.0 - jnp.power(1.0 + rate, -n_periods)
    standard_annuity = notional * rate / denominator
    return jnp.where(valid_mask, standard_annuity, result)


def discount_factor(
    rate: float,
    start: ActusDateTime,
    end: ActusDateTime,
    day_count_convention: DayCountConvention,
) -> float:
    """Calculate discount factor for a time period.

    Computes: DF = 1 / (1 + r * t)
    where t is the year fraction between start and end.

    Args:
        rate: Annual interest rate (e.g., 0.05 for 5%)
        start: Start date
        end: End date
        day_count_convention: Day count convention

    Returns:
        Discount factor

    Example:
        >>> start = ActusDateTime(2024, 1, 1, 0, 0, 0)
        >>> end = ActusDateTime(2024, 7, 1, 0, 0, 0)
        >>> df = discount_factor(0.05, start, end, DayCountConvention.AA)
        >>> abs(df - 0.9756) < 0.001  # Approximately 0.9756
        True

    References:
        ACTUS Technical Specification v1.1, Section 5.2
    """
    yf = year_fraction(start, end, day_count_convention)
    return 1.0 / (1.0 + rate * yf)


@jax.jit
def discount_factor_vectorized(
    rate: jnp.ndarray,
    year_fraction: jnp.ndarray,
) -> jnp.ndarray:
    """Vectorized discount factor calculation.

    Args:
        rate: Array of interest rates
        year_fraction: Array of year fractions

    Returns:
        Array of discount factors

    Example:
        >>> rates = jnp.array([0.05, 0.04, 0.06])
        >>> yfs = jnp.array([0.5, 1.0, 0.25])
        >>> dfs = discount_factor_vectorized(rates, yfs)

    Note:
        This function is JIT-compiled for performance.
    """
    return 1.0 / (1.0 + rate * year_fraction)


def compound_factor(
    rate: float,
    start: ActusDateTime,
    end: ActusDateTime,
    day_count_convention: DayCountConvention,
    compounding_frequency: int = 1,
) -> float:
    """Calculate compound factor for a time period.

    Computes: CF = (1 + r/m)^(m*t)
    where m is the compounding frequency and t is the year fraction.

    Args:
        rate: Annual interest rate (e.g., 0.05 for 5%)
        start: Start date
        end: End date
        day_count_convention: Day count convention
        compounding_frequency: Number of compounding periods per year (default 1)

    Returns:
        Compound factor

    Example:
        >>> start = ActusDateTime(2024, 1, 1, 0, 0, 0)
        >>> end = ActusDateTime(2025, 1, 1, 0, 0, 0)
        >>> # Annual compounding
        >>> cf = compound_factor(0.05, start, end, DayCountConvention.AA, 1)
        >>> abs(cf - 1.05) < 0.001
        True
        >>> # Monthly compounding
        >>> cf_monthly = compound_factor(0.05, start, end, DayCountConvention.AA, 12)
        >>> abs(cf_monthly - 1.05116) < 0.001
        True

    References:
        Standard financial mathematics
    """
    yf = year_fraction(start, end, day_count_convention)

    if compounding_frequency == 0:
        # Continuous compounding: e^(r*t)
        import math

        return math.exp(rate * yf)

    # Discrete compounding: (1 + r/m)^(m*t)
    return float((1.0 + rate / compounding_frequency) ** (compounding_frequency * yf))


@jax.jit
def compound_factor_vectorized(
    rate: jnp.ndarray,
    year_fraction: jnp.ndarray,
    compounding_frequency: jnp.ndarray,
) -> jnp.ndarray:
    """Vectorized compound factor calculation.

    Args:
        rate: Array of interest rates
        year_fraction: Array of year fractions
        compounding_frequency: Array of compounding frequencies

    Returns:
        Array of compound factors

    Example:
        >>> rates = jnp.array([0.05, 0.04])
        >>> yfs = jnp.array([1.0, 1.0])
        >>> freqs = jnp.array([1, 12])
        >>> cfs = compound_factor_vectorized(rates, yfs, freqs)

    Note:
        This function is JIT-compiled for performance.
        For continuous compounding (frequency=0), use a very large frequency instead.
    """
    # Handle continuous compounding (freq = 0) by using large number
    freq_adjusted = jnp.where(compounding_frequency == 0, 1e6, compounding_frequency)

    # (1 + r/m)^(m*t)
    base = 1.0 + rate / freq_adjusted
    exponent = freq_adjusted * year_fraction
    return jnp.power(base, exponent)


def present_value(
    cash_flows: list[float],
    dates: list[ActusDateTime],
    valuation_date: ActusDateTime,
    discount_rate: float,
    day_count_convention: DayCountConvention,
) -> float:
    """Calculate present value of a series of cash flows.

    Args:
        cash_flows: List of cash flow amounts
        dates: List of cash flow dates
        valuation_date: Date to discount to
        discount_rate: Annual discount rate
        day_count_convention: Day count convention

    Returns:
        Present value of all cash flows

    Example:
        >>> cfs = [100, 100, 100]
        >>> dates = [
        ...     ActusDateTime(2024, 1, 1, 0, 0, 0),
        ...     ActusDateTime(2024, 7, 1, 0, 0, 0),
        ...     ActusDateTime(2025, 1, 1, 0, 0, 0),
        ... ]
        >>> val_date = ActusDateTime(2024, 1, 1, 0, 0, 0)
        >>> pv = present_value(cfs, dates, val_date, 0.05, DayCountConvention.AA)
        >>> abs(pv - 295.14) < 1.0
        True

    References:
        Standard financial mathematics
    """
    if len(cash_flows) != len(dates):
        raise ValueError("cash_flows and dates must have the same length")

    total_pv = 0.0
    for cf, date in zip(cash_flows, dates, strict=True):
        if date >= valuation_date:
            df = discount_factor(discount_rate, valuation_date, date, day_count_convention)
            total_pv += cf * df
        else:
            # Cash flow in the past - compound forward
            cf_factor = compound_factor(discount_rate, date, valuation_date, day_count_convention)
            total_pv += cf * cf_factor

    return total_pv


@jax.jit
def present_value_vectorized(
    cash_flows: jnp.ndarray,
    year_fractions: jnp.ndarray,
    discount_rate: float,
) -> jnp.ndarray:
    """Vectorized present value calculation.

    Args:
        cash_flows: Array of cash flow amounts
        year_fractions: Array of year fractions from valuation date
        discount_rate: Discount rate

    Returns:
        Present value (scalar JAX array)

    Example:
        >>> cfs = jnp.array([100.0, 100.0, 100.0])
        >>> yfs = jnp.array([0.0, 0.5, 1.0])
        >>> pv = present_value_vectorized(cfs, yfs, 0.05)

    Note:
        This function is JIT-compiled for performance.
        Assumes all cash flows are in the future (year_fractions >= 0).
    """
    discount_factors = 1.0 / (1.0 + discount_rate * year_fractions)
    return jnp.sum(cash_flows * discount_factors)


def calculate_actus_annuity(
    start: ActusDateTime,
    pr_schedule: list[ActusDateTime],
    notional: float,
    accrued_interest: float,
    rate: float,
    day_count_convention: DayCountConvention,
) -> float:
    """Calculate annuity amount using ACTUS specification formula.

    Implements the ACTUS annuity formula from Section 3.8:
        A(s, T, n, a, r) = (n + a) / Σ[∏((1 + Y_i × r)^-1)]

    Where:
        s = start time
        T = maturity (last PR date)
        n = notional principal
        a = accrued interest
        r = nominal interest rate
        Y_i = year fraction for period i
        Σ = sum over all PR events
        ∏ = product up to each PR event

    This calculates the constant payment amount such that the total of all
    payments exactly amortizes the notional plus accrued interest.

    Args:
        start: Start time for calculation
        pr_schedule: List of principal redemption dates
        notional: Notional principal amount
        accrued_interest: Already accrued interest
        rate: Annual interest rate (e.g., 0.05 for 5%)
        day_count_convention: Day count convention for year fractions

    Returns:
        Annuity payment amount per period

    Example:
        >>> # $100,000 loan at 5% for 12 months
        >>> start = ActusDateTime(2024, 1, 15, 0, 0, 0)
        >>> pr_dates = [ActusDateTime(2024, i, 15, 0, 0, 0) for i in range(2, 14)]
        >>> amount = calculate_actus_annuity(
        ...     start, pr_dates, 100000.0, 0.0, 0.05, DayCountConvention.A360
        ... )
        >>> 8500 < amount < 8600  # Approximately $8,560
        True

    References:
        ACTUS Technical Specification v1.1, Section 3.8
    """
    if not pr_schedule:
        return 0.0

    if abs(rate) < 1e-10:
        # For zero or near-zero rates, payment is just (notional + accrued) / periods
        return (notional + accrued_interest) / len(pr_schedule)

    # Calculate the denominator: Σ[∏((1 + Y_i × r)^-1)]
    # This is the sum of discount factors for each period
    cumulative_discount = 0.0
    product_term = 1.0  # Running product of (1 + Y_i × r)

    prev_date = start
    for pr_date in pr_schedule:
        # Year fraction for this period
        yf = year_fraction(prev_date, pr_date, day_count_convention)

        # Update the cumulative product: (1 + Y_1 × r) × (1 + Y_2 × r) × ...
        product_term *= 1.0 + yf * rate

        # Add the discount factor for this period: 1 / ∏(1 + Y_i × r)
        cumulative_discount += 1.0 / product_term

        prev_date = pr_date

    # A = (n + a) / Σ[∏((1 + Y_i × r)^-1)]
    return (notional + accrued_interest) / cumulative_discount


@jax.jit
def calculate_actus_annuity_jax(
    year_fractions: jnp.ndarray,
    notional: float,
    accrued_interest: float,
    rate: float,
) -> float:
    """JAX-compiled version of ACTUS annuity calculation.

    Implements the ACTUS annuity formula:
        A(s, T, n, a, r) = (n + a) / Σ[∏((1 + Y_i × r)^-1)]

    Args:
        year_fractions: Array of year fractions for each period
        notional: Notional principal amount
        accrued_interest: Already accrued interest
        rate: Annual interest rate

    Returns:
        Annuity payment amount

    Example:
        >>> # 12 equal monthly periods (30/360 convention)
        >>> yfs = jnp.array([30/360] * 12)
        >>> amount = calculate_actus_annuity_jax(yfs, 100000.0, 0.0, 0.05)

    Note:
        This function is JIT-compiled for performance.

    References:
        ACTUS Technical Specification v1.1, Section 3.8
    """
    # Handle edge cases
    n_periods = year_fractions.shape[0]
    is_zero_rate = jnp.abs(rate) < 1e-10

    # For zero rate, simple division
    simple_payment = (notional + accrued_interest) / n_periods

    # Calculate cumulative discount factors
    # product_terms[i] = ∏_{j=0}^{i} (1 + Y_j × r)
    factors = 1.0 + year_fractions * rate
    product_terms = jnp.cumprod(factors)

    # discount_factors[i] = 1 / product_terms[i]
    discount_factors = 1.0 / product_terms

    # Sum of discount factors
    denominator = jnp.sum(discount_factors)

    # A = (n + a) / Σ[discount_factors]
    actus_payment = (notional + accrued_interest) / denominator

    # Return simple payment if rate is zero, otherwise ACTUS payment
    return jnp.where(is_zero_rate, simple_payment, actus_payment)
