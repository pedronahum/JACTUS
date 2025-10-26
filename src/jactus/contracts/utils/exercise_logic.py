"""Exercise logic utilities for options and derivatives.

This module provides JAX-compatible functions for calculating option intrinsic values,
making exercise decisions, and determining optimal exercise timing.

All functions are designed to be compatible with JAX's jit compilation and automatic
differentiation (grad, vmap, etc.) for efficient computation and Greeks calculation.

ACTUS Reference:
    ACTUS v1.1 Sections 7.15 (OPTNS) and 7.16 (FUTUR)

Example:
    >>> import jax.numpy as jnp
    >>> from jactus.contracts.utils import calculate_intrinsic_value
    >>>
    >>> # Calculate call option intrinsic value
    >>> spot_price = 105.0
    >>> strike = 100.0
    >>> intrinsic = calculate_intrinsic_value("C", spot_price, strike)
    >>> print(f"Call intrinsic value: {intrinsic}")  # 5.0
"""

from enum import Enum

import jax.numpy as jnp


class ExerciseDecision(Enum):
    """Exercise decision for an option or derivative.

    Values:
        EXERCISE: Exercise the option immediately
        HOLD: Continue holding the option (not in-the-money or American)
        EXPIRED: Option has expired out-of-the-money
    """

    EXERCISE = "EXERCISE"
    HOLD = "HOLD"
    EXPIRED = "EXPIRED"


def calculate_intrinsic_value(
    option_type: str,
    spot_price: float | jnp.ndarray,
    strike_1: float | jnp.ndarray,
    strike_2: float | jnp.ndarray | None = None,
) -> jnp.ndarray:
    """Calculate intrinsic value of an option.

    The intrinsic value is the immediate exercise value of the option,
    calculated as max(payoff, 0). This is JAX-compatible for automatic
    differentiation and JIT compilation.

    Args:
        option_type: Option type:
            - 'C' or 'CALL': Call option (right to buy)
            - 'P' or 'PUT': Put option (right to sell)
            - 'CP' or 'COLLAR': Collar (long call + short put, or vice versa)
        spot_price: Current market price of the underlying asset
        strike_1: Primary strike price (K for call/put, K1 for collar)
        strike_2: Secondary strike price (K2 for collar only)

    Returns:
        Intrinsic value as JAX array (always non-negative)

    Formulas:
        Call:   max(S - K, 0)
        Put:    max(K - S, 0)
        Collar: max(S - K1, 0) + max(K2 - S, 0)

    Example:
        >>> # Call option
        >>> iv = calculate_intrinsic_value("C", 105.0, 100.0)
        >>> print(iv)  # Array(5.0)
        >>>
        >>> # Put option
        >>> iv = calculate_intrinsic_value("P", 95.0, 100.0)
        >>> print(iv)  # Array(5.0)
        >>>
        >>> # Collar (long call at 100, short put at 90)
        >>> iv = calculate_intrinsic_value("CP", 105.0, 100.0, 90.0)
        >>> print(iv)  # Array(5.0)

    Note:
        This function uses JAX operations for compatibility with jit, grad, and vmap.
        For Greeks calculation, use jax.grad on this function.
    """
    # Convert to JAX arrays
    s = jnp.asarray(spot_price, dtype=jnp.float32)
    k1 = jnp.asarray(strike_1, dtype=jnp.float32)

    # Normalize option type
    opt_type = option_type.upper()

    if opt_type in ("C", "CALL"):
        # Call option: right to buy at strike
        # Intrinsic value = max(S - K, 0)
        return jnp.maximum(s - k1, 0.0)

    if opt_type in ("P", "PUT"):
        # Put option: right to sell at strike
        # Intrinsic value = max(K - S, 0)
        return jnp.maximum(k1 - s, 0.0)

    if opt_type in ("CP", "COLLAR"):
        # Collar: combination of call and put
        # Intrinsic value = max(S - K1, 0) + max(K2 - S, 0)
        if strike_2 is None:
            raise ValueError("strike_2 required for collar option (CP)")

        k2 = jnp.asarray(strike_2, dtype=jnp.float32)

        # Long call at K1 + short put at K2 (or vice versa)
        call_value = jnp.maximum(s - k1, 0.0)
        put_value = jnp.maximum(k2 - s, 0.0)

        return call_value + put_value

    raise ValueError(
        f"Unknown option type: {option_type}. "
        f"Expected 'C'/'CALL', 'P'/'PUT', or 'CP'/'COLLAR'"
    )


def should_exercise(
    intrinsic_value: float | jnp.ndarray,
    threshold: float = 0.0,
) -> jnp.ndarray:
    """Determine if an option should be exercised based on intrinsic value.

    For simple exercise logic, an option should be exercised if its intrinsic
    value exceeds a threshold (typically 0 for European options, possibly higher
    for American options to account for time value).

    Args:
        intrinsic_value: Current intrinsic value of the option
        threshold: Minimum value to trigger exercise (default: 0.0)

    Returns:
        Boolean JAX array: True if should exercise, False otherwise

    Example:
        >>> intrinsic = jnp.array(5.0)
        >>> should_ex = should_exercise(intrinsic, threshold=0.0)
        >>> print(should_ex)  # True
        >>>
        >>> intrinsic = jnp.array(0.0)
        >>> should_ex = should_exercise(intrinsic, threshold=0.0)
        >>> print(should_ex)  # False

    Note:
        This is a simple exercise rule. For American options, optimal exercise
        may require comparing intrinsic value to continuation value (early exercise
        premium). For now, we use a simple threshold-based rule.
    """
    iv = jnp.asarray(intrinsic_value, dtype=jnp.float32)
    thresh = jnp.asarray(threshold, dtype=jnp.float32)

    return iv > thresh


def calculate_option_greeks(
    option_type: str,
    spot_price: float,
    strike: float,
    volatility: float,
    time_to_maturity: float,
    risk_free_rate: float = 0.0,
) -> dict[str, float]:
    """Calculate option Greeks using JAX automatic differentiation.

    This function demonstrates how to use JAX's grad to compute option sensitivities
    (Greeks) from the intrinsic value function. For more accurate Greeks, a full
    Black-Scholes implementation would be needed.

    Args:
        option_type: 'C' for call, 'P' for put
        spot_price: Current underlying price
        strike: Strike price
        volatility: Implied volatility (not used in intrinsic value, placeholder)
        time_to_maturity: Time to expiration in years (not used, placeholder)
        risk_free_rate: Risk-free interest rate (not used, placeholder)

    Returns:
        Dictionary containing Greeks:
            - delta: ∂V/∂S (sensitivity to underlying price)
            - intrinsic: Current intrinsic value

    Note:
        This is a simplified implementation using intrinsic value only.
        For production use, implement full Black-Scholes pricing with time value.

    Example:
        >>> greeks = calculate_option_greeks("C", 105.0, 100.0, 0.2, 1.0, 0.05)
        >>> print(f"Delta: {greeks['delta']}")
        >>> print(f"Intrinsic: {greeks['intrinsic']}")
    """
    # This is a simplified example showing how to use JAX for Greeks
    # For production, you'd implement full Black-Scholes pricing

    # Calculate intrinsic value
    intrinsic = calculate_intrinsic_value(option_type, spot_price, strike)

    # Use JAX grad to calculate delta (∂V/∂S)
    # For intrinsic value, delta is 1 if in-the-money, 0 otherwise
    # (This is a step function, so grad gives 0 everywhere except at strike)

    from jax import grad

    # Define a smoothed payoff for better gradients
    def smooth_payoff(s: jnp.ndarray) -> jnp.ndarray:
        return calculate_intrinsic_value(option_type, s, strike)

    # Calculate delta
    delta_fn = grad(smooth_payoff)
    delta = delta_fn(jnp.array(spot_price, dtype=jnp.float32))

    return {
        "delta": float(delta),
        "intrinsic": float(intrinsic),
    }
