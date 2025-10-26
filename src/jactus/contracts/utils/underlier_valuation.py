"""Underlier valuation utilities for derivative contracts.

This module provides functions for valuing underlier assets referenced by
derivative contracts (options, futures, swaps, etc.).

Underliers can be:
- Market objects (stocks, commodities, indices) - observed from market data
- Other contracts (bonds, loans) - valued via simulation or canonical payoff

ACTUS Reference:
    ACTUS v1.1 Sections on Derivative Contracts (7.11-7.18)

Example:
    >>> from jactus.contracts.utils import get_underlier_market_value
    >>> from jactus.observers import ConstantRiskFactorObserver
    >>>
    >>> # Get stock price from market
    >>> rf_obs = ConstantRiskFactorObserver(constant_value=150.0)
    >>> value = get_underlier_market_value("AAPL", time, rf_obs)
    >>> print(f"Stock price: {value}")  # 150.0
"""

from typing import Any

import jax.numpy as jnp

from jactus.core import ActusDateTime
from jactus.observers import RiskFactorObserver


def get_underlier_market_value(
    underlier_reference: str | Any,
    time: ActusDateTime,
    risk_factor_observer: RiskFactorObserver,
) -> jnp.ndarray:
    """Get market value of an underlier asset at a given time.

    This function queries the risk factor observer to get the current market
    price of the underlier. The underlier can be:
    - A market object code (e.g., "AAPL", "GLD", "SPX")
    - A contract reference (for contract-based underliers)

    Args:
        underlier_reference: Reference to the underlier asset
            - str: Market object code (ticker symbol, index code, etc.)
            - Contract: Reference to another contract (for future implementation)
        time: Time at which to observe the market value
        risk_factor_observer: Observer for market data

    Returns:
        Market value as JAX array

    Example:
        >>> # Get stock price
        >>> rf_obs = ConstantRiskFactorObserver(constant_value=150.0)
        >>> value = get_underlier_market_value("AAPL", time, rf_obs)
        >>> print(value)  # Array(150.0)
        >>>
        >>> # Get commodity price
        >>> value = get_underlier_market_value("GC", time, rf_obs)  # Gold
        >>> print(value)  # Array(1800.0)

    Note:
        For contract-based underliers, this function would need to simulate
        the underlier contract and calculate its market value. This is a
        simplified implementation for market object underliers only.
    """
    # For string references (market objects), observe from risk factor
    if isinstance(underlier_reference, str):
        # Query risk factor observer for market price
        market_value = risk_factor_observer.observe_risk_factor(underlier_reference, time)
        return jnp.asarray(market_value, dtype=jnp.float32)

    # For contract references, we would need to:
    # 1. Simulate the contract (or use canonical payoff)
    # 2. Calculate its present value
    # 3. Return the market value
    #
    # This is complex and depends on the contract type, so we'll implement
    # it when needed for SWAPS and other composite contracts.

    raise NotImplementedError(
        f"Underlier valuation for type {type(underlier_reference)} not yet implemented. "
        f"Currently only market object codes (strings) are supported."
    )


def get_underlier_price_path(
    underlier_reference: str,
    times: list[ActusDateTime],
    risk_factor_observer: RiskFactorObserver,
) -> jnp.ndarray:
    """Get price path of an underlier over multiple time points.

    This is useful for American options where we need to check exercise
    decisions at multiple points in time.

    Args:
        underlier_reference: Market object code for the underlier
        times: List of times at which to observe prices
        risk_factor_observer: Observer for market data

    Returns:
        Array of prices at each time point

    Example:
        >>> times = [
        ...     ActusDateTime(2024, 1, 1, 0, 0, 0),
        ...     ActusDateTime(2024, 2, 1, 0, 0, 0),
        ...     ActusDateTime(2024, 3, 1, 0, 0, 0),
        ... ]
        >>> prices = get_underlier_price_path("AAPL", times, rf_obs)
        >>> print(prices)  # Array([150.0, 152.0, 155.0])
    """
    prices = []
    for t in times:
        price = get_underlier_market_value(underlier_reference, t, risk_factor_observer)
        prices.append(price)

    return jnp.array(prices, dtype=jnp.float32)


def calculate_forward_price(
    spot_price: float | jnp.ndarray,
    risk_free_rate: float | jnp.ndarray,
    time_to_maturity: float | jnp.ndarray,
    dividend_yield: float | jnp.ndarray = 0.0,
) -> jnp.ndarray:
    """Calculate theoretical forward price of an asset.

    Uses cost-of-carry model to calculate forward price:
        F = S × exp((r - q) × T)

    Where:
        S: Spot price
        r: Risk-free rate
        q: Dividend yield (or convenience yield for commodities)
        T: Time to maturity

    Args:
        spot_price: Current spot price of the asset
        risk_free_rate: Risk-free interest rate (annualized)
        time_to_maturity: Time to maturity in years
        dividend_yield: Dividend yield (default: 0.0)

    Returns:
        Theoretical forward price

    Example:
        >>> # Stock forward price (6-month forward)
        >>> spot = 100.0
        >>> rate = 0.05  # 5%
        >>> time = 0.5   # 6 months
        >>> div_yield = 0.02  # 2% dividend yield
        >>> fwd = calculate_forward_price(spot, rate, time, div_yield)
        >>> print(f"Forward price: {fwd}")  # ~101.5

    Note:
        This is used for validating FUTUR contract pricing and can be
        used in arbitrage analysis.
    """
    s = jnp.asarray(spot_price, dtype=jnp.float32)
    r = jnp.asarray(risk_free_rate, dtype=jnp.float32)
    t = jnp.asarray(time_to_maturity, dtype=jnp.float32)
    q = jnp.asarray(dividend_yield, dtype=jnp.float32)

    # Forward price: F = S × exp((r - q) × T)
    return s * jnp.exp((r - q) * t)


def calculate_present_value(
    future_cashflow: float | jnp.ndarray,
    discount_rate: float | jnp.ndarray,
    time_to_payment: float | jnp.ndarray,
) -> jnp.ndarray:
    """Calculate present value of a future cashflow.

    Uses simple exponential discounting:
        PV = CF × exp(-r × T)

    Args:
        future_cashflow: Cashflow amount at future time
        discount_rate: Discount rate (annualized)
        time_to_payment: Time to payment in years

    Returns:
        Present value of the cashflow

    Example:
        >>> # Present value of $100 in 1 year at 5%
        >>> pv = calculate_present_value(100.0, 0.05, 1.0)
        >>> print(f"PV: {pv}")  # ~95.12
    """
    cf = jnp.asarray(future_cashflow, dtype=jnp.float32)
    r = jnp.asarray(discount_rate, dtype=jnp.float32)
    t = jnp.asarray(time_to_payment, dtype=jnp.float32)

    # Present value: PV = CF × exp(-r × T)
    return cf * jnp.exp(-r * t)
