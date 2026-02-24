"""Risk factor observer for ACTUS contracts.

This module implements the Risk Factor Observer (O_rf) framework, which provides
access to market data and risk factors needed for contract valuation.

The observer has two key methods:
- O_rf(i, t, S, M): Observe risk factor i at time t
- O_ev(i, k, t, S, M): Observe event-related data

References:
    ACTUS v1.1 Section 2.9 - Risk Factor Observer
"""

from __future__ import annotations

import bisect
import math
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import jax.numpy as jnp

if TYPE_CHECKING:
    from jactus.core import ActusDateTime, ContractAttributes, ContractState
    from jactus.core.types import EventType


@runtime_checkable
class RiskFactorObserver(Protocol):
    """Protocol for risk factor observers.

    The risk factor observer provides access to market data and risk factors
    needed for contract calculations. It abstracts away the data source
    (historical data, real-time feeds, simulations, etc.).

    All implementations must be JAX-compatible where possible.
    """

    def observe_risk_factor(
        self,
        identifier: str,
        time: ActusDateTime,
        state: ContractState | None = None,
        attributes: ContractAttributes | None = None,
    ) -> jnp.ndarray:
        """Observe a risk factor at a specific time.

        This is the O_rf(i, t, S, M) function from ACTUS specification.

        Args:
            identifier: Risk factor identifier (e.g., "USD/EUR", "LIBOR-3M")
            time: Time at which to observe the risk factor
            state: Current contract state (optional, for state-dependent factors)
            attributes: Contract attributes (optional, for contract-dependent factors)

        Returns:
            Risk factor value as JAX array

        Example:
            >>> observer = DictRiskFactorObserver({"USD/EUR": 1.18})
            >>> fx_rate = observer.observe_risk_factor("USD/EUR", time)
            >>> # Returns 1.18
        """
        ...

    def observe_event(
        self,
        identifier: str,
        event_type: EventType,
        time: ActusDateTime,
        state: ContractState | None = None,
        attributes: ContractAttributes | None = None,
    ) -> Any:
        """Observe event-related data.

        This is the O_ev(i, k, t, S, M) function from ACTUS specification,
        where k is the event type.

        Args:
            identifier: Event data identifier
            event_type: Type of event
            time: Time at which to observe
            state: Current contract state (optional)
            attributes: Contract attributes (optional)

        Returns:
            Event-related data (type depends on identifier)

        Example:
            >>> observer = DictRiskFactorObserver(...)
            >>> rate = observer.observe_event("RESET_RATE", EventType.RR, time)
        """
        ...


class BaseRiskFactorObserver(ABC):
    """Base class for risk factor observers with common functionality.

    This class provides a framework for implementing risk factor observers
    with caching, error handling, and JAX compatibility.
    """

    def __init__(self, name: str | None = None):
        """Initialize base risk factor observer.

        Args:
            name: Optional name for this observer (for debugging)
        """
        self.name = name or self.__class__.__name__

    @abstractmethod
    def _get_risk_factor(
        self,
        identifier: str,
        time: ActusDateTime,
        state: ContractState | None,
        attributes: ContractAttributes | None,
    ) -> jnp.ndarray:
        """Get risk factor value from underlying data source.

        This method must be implemented by subclasses to define how
        risk factors are retrieved.

        Args:
            identifier: Risk factor identifier
            time: Time at which to observe
            state: Current contract state (optional)
            attributes: Contract attributes (optional)

        Returns:
            Risk factor value as JAX array

        Raises:
            KeyError: If risk factor is not found
            ValueError: If risk factor data is invalid
        """
        ...

    @abstractmethod
    def _get_event_data(
        self,
        identifier: str,
        event_type: EventType,
        time: ActusDateTime,
        state: ContractState | None,
        attributes: ContractAttributes | None,
    ) -> Any:
        """Get event-related data from underlying data source.

        This method must be implemented by subclasses to define how
        event data is retrieved.

        Args:
            identifier: Event data identifier
            event_type: Type of event
            time: Time at which to observe
            state: Current contract state (optional)
            attributes: Contract attributes (optional)

        Returns:
            Event-related data

        Raises:
            KeyError: If event data is not found
            ValueError: If event data is invalid
        """
        ...

    def observe_risk_factor(
        self,
        identifier: str,
        time: ActusDateTime,
        state: ContractState | None = None,
        attributes: ContractAttributes | None = None,
    ) -> jnp.ndarray:
        """Observe a risk factor at a specific time.

        This method wraps _get_risk_factor with error handling and logging.

        Args:
            identifier: Risk factor identifier
            time: Time at which to observe
            state: Current contract state (optional)
            attributes: Contract attributes (optional)

        Returns:
            Risk factor value as JAX array

        Raises:
            KeyError: If risk factor is not found
        """
        return self._get_risk_factor(identifier, time, state, attributes)

    def observe_event(
        self,
        identifier: str,
        event_type: EventType,
        time: ActusDateTime,
        state: ContractState | None = None,
        attributes: ContractAttributes | None = None,
    ) -> Any:
        """Observe event-related data.

        This method wraps _get_event_data with error handling and logging.

        Args:
            identifier: Event data identifier
            event_type: Type of event
            time: Time at which to observe
            state: Current contract state (optional)
            attributes: Contract attributes (optional)

        Returns:
            Event-related data
        """
        return self._get_event_data(identifier, event_type, time, state, attributes)


class ConstantRiskFactorObserver(BaseRiskFactorObserver):
    """Risk factor observer that returns constant values.

    This is useful for testing or for contracts with fixed risk factors.

    Example:
        >>> observer = ConstantRiskFactorObserver(1.0)
        >>> rate = observer.observe_risk_factor("ANY_RATE", time)
        >>> # Always returns 1.0
    """

    def __init__(self, constant_value: float, name: str | None = None):
        """Initialize constant risk factor observer.

        Args:
            constant_value: The constant value to return for all risk factors
            name: Optional name for this observer
        """
        super().__init__(name)
        self.constant_value = jnp.array(constant_value, dtype=jnp.float32)

    def _get_risk_factor(
        self,
        identifier: str,  # noqa: ARG002
        time: ActusDateTime,  # noqa: ARG002
        state: ContractState | None,  # noqa: ARG002
        attributes: ContractAttributes | None,  # noqa: ARG002
    ) -> jnp.ndarray:
        """Return constant value for any risk factor.

        Args:
            identifier: Risk factor identifier (ignored)
            time: Time at which to observe (ignored)
            state: Current contract state (ignored)
            attributes: Contract attributes (ignored)

        Returns:
            Constant value as JAX array
        """
        return self.constant_value

    def _get_event_data(
        self,
        identifier: str,  # noqa: ARG002
        event_type: EventType,
        time: ActusDateTime,  # noqa: ARG002
        state: ContractState | None,  # noqa: ARG002
        attributes: ContractAttributes | None,  # noqa: ARG002
    ) -> Any:
        """Return constant value for any event data.

        Args:
            identifier: Event data identifier (ignored)
            event_type: Type of event (ignored)
            time: Time at which to observe (ignored)
            state: Current contract state (ignored)
            attributes: Contract attributes (ignored)

        Returns:
            Constant value as JAX array
        """
        return self.constant_value


class DictRiskFactorObserver(BaseRiskFactorObserver):
    """Risk factor observer backed by a dictionary.

    This is useful for testing or for simple scenarios with a fixed set
    of risk factors.

    Example:
        >>> data = {
        ...     "USD/EUR": 1.18,
        ...     "LIBOR-3M": 0.02,
        ... }
        >>> observer = DictRiskFactorObserver(data)
        >>> fx_rate = observer.observe_risk_factor("USD/EUR", time)
        >>> # Returns 1.18
    """

    def __init__(
        self,
        risk_factors: dict[str, float],
        event_data: dict[str, Any] | None = None,
        name: str | None = None,
    ):
        """Initialize dictionary-backed risk factor observer.

        Args:
            risk_factors: Dictionary mapping risk factor identifiers to values
            event_data: Dictionary mapping event data identifiers to values (optional)
            name: Optional name for this observer
        """
        super().__init__(name)
        # Convert all values to JAX arrays
        self.risk_factors = {k: jnp.array(v, dtype=jnp.float32) for k, v in risk_factors.items()}
        self.event_data = event_data or {}

    def _get_risk_factor(
        self,
        identifier: str,  # noqa: ARG002
        time: ActusDateTime,  # noqa: ARG002
        state: ContractState | None,  # noqa: ARG002
        attributes: ContractAttributes | None,  # noqa: ARG002
    ) -> jnp.ndarray:
        """Get risk factor value from dictionary.

        Args:
            identifier: Risk factor identifier
            time: Time at which to observe (ignored for this implementation)
            state: Current contract state (ignored for this implementation)
            attributes: Contract attributes (ignored for this implementation)

        Returns:
            Risk factor value as JAX array

        Raises:
            KeyError: If risk factor identifier is not found
        """
        if identifier not in self.risk_factors:
            raise KeyError(f"Risk factor '{identifier}' not found in observer '{self.name}'")
        return self.risk_factors[identifier]

    def _get_event_data(
        self,
        identifier: str,  # noqa: ARG002
        event_type: EventType,
        time: ActusDateTime,  # noqa: ARG002
        state: ContractState | None,  # noqa: ARG002
        attributes: ContractAttributes | None,  # noqa: ARG002
    ) -> Any:
        """Get event data from dictionary.

        Args:
            identifier: Event data identifier
            event_type: Type of event (ignored for this implementation)
            time: Time at which to observe (ignored for this implementation)
            state: Current contract state (ignored for this implementation)
            attributes: Contract attributes (ignored for this implementation)

        Returns:
            Event-related data

        Raises:
            KeyError: If event data identifier is not found
        """
        if identifier not in self.event_data:
            raise KeyError(f"Event data '{identifier}' not found in observer '{self.name}'")
        return self.event_data[identifier]

    def add_risk_factor(self, identifier: str, value: float) -> None:
        """Add or update a risk factor.

        Args:
            identifier: Risk factor identifier
            value: Risk factor value
        """
        self.risk_factors[identifier] = jnp.array(value, dtype=jnp.float32)

    def add_event_data(self, identifier: str, value: Any) -> None:
        """Add or update event data.

        Args:
            identifier: Event data identifier
            value: Event data value
        """
        self.event_data[identifier] = value


class TimeSeriesRiskFactorObserver(BaseRiskFactorObserver):
    """Risk factor observer backed by time series data with interpolation.

    Maps identifiers to time-ordered sequences of (ActusDateTime, float) pairs.
    Supports step (piecewise constant) and linear interpolation, with flat or
    raising extrapolation behavior.

    Example:
        >>> from jactus.core import ActusDateTime
        >>> ts = {
        ...     "LIBOR-3M": [
        ...         (ActusDateTime(2024, 1, 1), 0.04),
        ...         (ActusDateTime(2024, 7, 1), 0.045),
        ...         (ActusDateTime(2025, 1, 1), 0.05),
        ...     ]
        ... }
        >>> observer = TimeSeriesRiskFactorObserver(ts)
        >>> rate = observer.observe_risk_factor(
        ...     "LIBOR-3M", ActusDateTime(2024, 4, 1)
        ... )
        >>> # Returns 0.04 (step interpolation: last known value)
    """

    def __init__(
        self,
        risk_factors: dict[str, list[tuple[ActusDateTime, float]]],
        event_data: dict[str, list[tuple[ActusDateTime, Any]]] | None = None,
        interpolation: str = "step",
        extrapolation: str = "flat",
        name: str | None = None,
    ):
        """Initialize time series risk factor observer.

        Args:
            risk_factors: Dict mapping identifiers to time-value pairs.
            event_data: Optional dict mapping identifiers to time-value pairs for events.
            interpolation: Interpolation method: "step" (piecewise constant) or "linear".
            extrapolation: Extrapolation method: "flat" (nearest endpoint) or "raise" (KeyError).
            name: Optional name for this observer.
        """
        if interpolation not in ("step", "linear"):
            raise ValueError(f"interpolation must be 'step' or 'linear', got '{interpolation}'")
        if extrapolation not in ("flat", "raise"):
            raise ValueError(f"extrapolation must be 'flat' or 'raise', got '{extrapolation}'")
        super().__init__(name)
        self.interpolation = interpolation
        self.extrapolation = extrapolation
        # Sort each series by time and convert values to JAX arrays
        self._risk_factor_series: dict[str, list[tuple[ActusDateTime, jnp.ndarray]]] = {}
        for identifier, series in risk_factors.items():
            sorted_series = sorted(series, key=lambda x: x[0])
            self._risk_factor_series[identifier] = [
                (t, jnp.array(v, dtype=jnp.float32)) for t, v in sorted_series
            ]
        self._event_data_series: dict[str, list[tuple[ActusDateTime, Any]]] = {}
        if event_data:
            for identifier, series in event_data.items():
                self._event_data_series[identifier] = sorted(series, key=lambda x: x[0])

    def _interpolate(
        self,
        series: list[tuple[ActusDateTime, jnp.ndarray]],
        time: ActusDateTime,
        identifier: str,
    ) -> jnp.ndarray:
        """Find interpolated value in a sorted time series."""
        if not series:
            raise KeyError(f"Empty time series for '{identifier}' in observer '{self.name}'")

        times = [entry[0] for entry in series]

        # Before first point
        if time < times[0]:
            if self.extrapolation == "raise":
                raise KeyError(
                    f"Time {time} is before first observation for '{identifier}' "
                    f"in observer '{self.name}'"
                )
            return series[0][1]

        # At or after last point
        if time >= times[-1]:
            if len(times) > 1 and time > times[-1] and self.extrapolation == "raise":
                raise KeyError(
                    f"Time {time} is after last observation for '{identifier}' "
                    f"in observer '{self.name}'"
                )
            return series[-1][1]

        # Find interval using binary search
        idx = bisect.bisect_right(times, time) - 1

        if self.interpolation == "step":
            return series[idx][1]

        # Linear interpolation
        t0, v0 = series[idx]
        t1, v1 = series[idx + 1]
        days_total = t0.days_between(t1)
        if days_total == 0:
            return v0
        days_elapsed = t0.days_between(time)
        frac = days_elapsed / days_total
        return jnp.array(float(v0) + frac * (float(v1) - float(v0)), dtype=jnp.float32)

    def _get_risk_factor(
        self,
        identifier: str,
        time: ActusDateTime,
        state: ContractState | None,  # noqa: ARG002
        attributes: ContractAttributes | None,  # noqa: ARG002
    ) -> jnp.ndarray:
        """Get interpolated risk factor value from time series.

        Args:
            identifier: Risk factor identifier.
            time: Time at which to observe.
            state: Current contract state (ignored).
            attributes: Contract attributes (ignored).

        Returns:
            Interpolated risk factor value as JAX array.

        Raises:
            KeyError: If identifier not found or time out of range with raise extrapolation.
        """
        if identifier not in self._risk_factor_series:
            raise KeyError(f"Risk factor '{identifier}' not found in observer '{self.name}'")
        return self._interpolate(self._risk_factor_series[identifier], time, identifier)

    def _get_event_data(
        self,
        identifier: str,
        event_type: EventType,
        time: ActusDateTime,
        state: ContractState | None,  # noqa: ARG002
        attributes: ContractAttributes | None,  # noqa: ARG002
    ) -> Any:
        """Get interpolated event data from time series.

        Args:
            identifier: Event data identifier.
            event_type: Type of event (ignored).
            time: Time at which to observe.
            state: Current contract state (ignored).
            attributes: Contract attributes (ignored).

        Returns:
            Interpolated event data value.

        Raises:
            KeyError: If identifier not found.
        """
        if identifier not in self._event_data_series:
            raise KeyError(f"Event data '{identifier}' not found in observer '{self.name}'")
        series = self._event_data_series[identifier]
        if not series:
            raise KeyError(f"Empty event data series for '{identifier}' in observer '{self.name}'")
        times = [entry[0] for entry in series]
        if time < times[0]:
            if self.extrapolation == "raise":
                raise KeyError(
                    f"Time {time} is before first observation for event data '{identifier}'"
                )
            return series[0][1]
        if time >= times[-1]:
            if len(times) > 1 and time > times[-1] and self.extrapolation == "raise":
                raise KeyError(
                    f"Time {time} is after last observation for event data '{identifier}'"
                )
            return series[-1][1]
        idx = bisect.bisect_right(times, time) - 1
        return series[idx][1]


class CurveRiskFactorObserver(BaseRiskFactorObserver):
    """Risk factor observer for yield/rate curves.

    Maps identifiers to tenor-rate curves where each curve is a list of
    (tenor_years, rate) pairs. Given an observation time, the observer
    computes the tenor from the reference date and interpolates the curve.

    Example:
        >>> from jactus.core import ActusDateTime
        >>> curve = {
        ...     "USD-YIELD": [
        ...         (0.25, 0.03),   # 3-month rate
        ...         (1.0, 0.04),    # 1-year rate
        ...         (5.0, 0.05),    # 5-year rate
        ...     ]
        ... }
        >>> observer = CurveRiskFactorObserver(
        ...     curves=curve,
        ...     reference_date=ActusDateTime(2024, 1, 1),
        ... )
        >>> rate = observer.observe_risk_factor(
        ...     "USD-YIELD", ActusDateTime(2024, 7, 1)
        ... )
    """

    def __init__(
        self,
        curves: dict[str, list[tuple[float, float]]],
        reference_date: ActusDateTime | None = None,
        interpolation: str = "linear",
        name: str | None = None,
    ):
        """Initialize curve risk factor observer.

        Args:
            curves: Dict mapping identifiers to lists of (tenor_years, rate) pairs.
            reference_date: Base date for tenor calculation. Falls back to
                attributes.status_date if not set.
            interpolation: Interpolation method: "linear" or "log_linear".
            name: Optional name for this observer.
        """
        if interpolation not in ("linear", "log_linear"):
            raise ValueError(
                f"interpolation must be 'linear' or 'log_linear', got '{interpolation}'"
            )
        super().__init__(name)
        self.reference_date = reference_date
        self.interpolation = interpolation
        # Sort curves by tenor, convert rates to JAX arrays
        self._curves: dict[str, list[tuple[float, jnp.ndarray]]] = {}
        for identifier, curve in curves.items():
            sorted_curve = sorted(curve, key=lambda x: x[0])
            if interpolation == "log_linear":
                for tenor, rate in sorted_curve:
                    if rate <= 0:
                        raise ValueError(
                            f"log_linear interpolation requires positive rates, "
                            f"got {rate} at tenor {tenor} for '{identifier}'"
                        )
            self._curves[identifier] = [
                (tenor, jnp.array(rate, dtype=jnp.float32)) for tenor, rate in sorted_curve
            ]

    def _get_risk_factor(
        self,
        identifier: str,
        time: ActusDateTime,
        state: ContractState | None,  # noqa: ARG002
        attributes: ContractAttributes | None,
    ) -> jnp.ndarray:
        """Get interpolated rate from yield curve.

        Args:
            identifier: Curve identifier.
            time: Time at which to observe (used to compute tenor from reference date).
            state: Current contract state (ignored).
            attributes: Contract attributes (used for status_date fallback).

        Returns:
            Interpolated rate as JAX array.

        Raises:
            KeyError: If identifier not found.
            ValueError: If no reference date available.
        """
        if identifier not in self._curves:
            raise KeyError(f"Curve '{identifier}' not found in observer '{self.name}'")

        ref_date = self.reference_date
        if ref_date is None and attributes is not None:
            ref_date = attributes.status_date
        if ref_date is None:
            raise ValueError(
                "CurveRiskFactorObserver requires reference_date or attributes.status_date"
            )

        tenor = ref_date.days_between(time) / 365.25
        curve = self._curves[identifier]

        if not curve:
            raise KeyError(f"Empty curve for '{identifier}' in observer '{self.name}'")

        tenors = [entry[0] for entry in curve]

        # Extrapolation: flat
        if tenor <= tenors[0]:
            return curve[0][1]
        if tenor >= tenors[-1]:
            return curve[-1][1]

        # Find interval
        idx = bisect.bisect_right(tenors, tenor) - 1
        t0, r0 = curve[idx]
        t1, r1 = curve[idx + 1]

        if t1 == t0:
            return r0

        frac = (tenor - t0) / (t1 - t0)

        if self.interpolation == "linear":
            return jnp.array(float(r0) + frac * (float(r1) - float(r0)), dtype=jnp.float32)

        # Log-linear interpolation
        log_r = math.log(float(r0)) + frac * (math.log(float(r1)) - math.log(float(r0)))
        return jnp.array(math.exp(log_r), dtype=jnp.float32)

    def _get_event_data(
        self,
        identifier: str,
        event_type: EventType,
        time: ActusDateTime,  # noqa: ARG002
        state: ContractState | None,  # noqa: ARG002
        attributes: ContractAttributes | None,  # noqa: ARG002
    ) -> Any:
        """Curve observer does not support event data.

        Raises:
            KeyError: Always, as curves don't provide event data.
        """
        raise KeyError(
            f"CurveRiskFactorObserver does not support event data lookup for '{identifier}'"
        )


class CallbackRiskFactorObserver(BaseRiskFactorObserver):
    """Risk factor observer that delegates to user-provided callables.

    Provides maximum flexibility by allowing arbitrary Python functions
    to produce risk factor values.

    Example:
        >>> import math
        >>> def my_rate(identifier: str, time: ActusDateTime) -> float:
        ...     years = ActusDateTime(2024, 1, 1).years_between(time)
        ...     return 0.03 + 0.01 * math.log(1 + max(years, 0))
        ...
        >>> observer = CallbackRiskFactorObserver(callback=my_rate)
        >>> rate = observer.observe_risk_factor("ANY", ActusDateTime(2025, 1, 1))
    """

    def __init__(
        self,
        callback: Callable[[str, ActusDateTime], float],
        event_callback: Callable[[str, EventType, ActusDateTime], Any] | None = None,
        name: str | None = None,
    ):
        """Initialize callback risk factor observer.

        Args:
            callback: Function taking (identifier, time) and returning a float.
            event_callback: Optional function taking (identifier, event_type, time)
                and returning event data.
            name: Optional name for this observer.
        """
        super().__init__(name)
        self.callback = callback
        self.event_callback = event_callback

    def _get_risk_factor(
        self,
        identifier: str,
        time: ActusDateTime,
        state: ContractState | None,  # noqa: ARG002
        attributes: ContractAttributes | None,  # noqa: ARG002
    ) -> jnp.ndarray:
        """Get risk factor value from callback.

        Args:
            identifier: Risk factor identifier.
            time: Time at which to observe.
            state: Current contract state (ignored).
            attributes: Contract attributes (ignored).

        Returns:
            Callback result as JAX array.
        """
        result = self.callback(identifier, time)
        return jnp.array(result, dtype=jnp.float32)

    def _get_event_data(
        self,
        identifier: str,
        event_type: EventType,
        time: ActusDateTime,
        state: ContractState | None,  # noqa: ARG002
        attributes: ContractAttributes | None,  # noqa: ARG002
    ) -> Any:
        """Get event data from callback.

        Args:
            identifier: Event data identifier.
            event_type: Type of event.
            time: Time at which to observe.
            state: Current contract state (ignored).
            attributes: Contract attributes (ignored).

        Returns:
            Event callback result.

        Raises:
            KeyError: If no event callback is configured.
        """
        if self.event_callback is None:
            raise KeyError(
                f"No event callback configured in observer '{self.name}' "
                f"for identifier '{identifier}'"
            )
        return self.event_callback(identifier, event_type, time)


class CompositeRiskFactorObserver(BaseRiskFactorObserver):
    """Risk factor observer that chains multiple observers with fallback.

    Tries each observer in order and returns the first successful result.
    If an observer raises KeyError, the next one is tried. Other exceptions
    propagate immediately.

    Example:
        >>> ts_observer = TimeSeriesRiskFactorObserver({"LIBOR-3M": [...]})
        >>> fallback = ConstantRiskFactorObserver(0.0)
        >>> composite = CompositeRiskFactorObserver([ts_observer, fallback])
        >>> # Uses ts_observer for "LIBOR-3M", falls back to constant for others
    """

    def __init__(
        self,
        observers: list[RiskFactorObserver],
        name: str | None = None,
    ):
        """Initialize composite risk factor observer.

        Args:
            observers: List of observers to try in order. Must not be empty.
            name: Optional name for this observer.

        Raises:
            ValueError: If observers list is empty.
        """
        if not observers:
            raise ValueError("observers list must not be empty")
        super().__init__(name)
        self.observers = observers

    def _get_risk_factor(
        self,
        identifier: str,
        time: ActusDateTime,
        state: ContractState | None,
        attributes: ContractAttributes | None,
    ) -> jnp.ndarray:
        """Try each observer in order, return first successful result.

        Args:
            identifier: Risk factor identifier.
            time: Time at which to observe.
            state: Current contract state.
            attributes: Contract attributes.

        Returns:
            Risk factor value from first matching observer.

        Raises:
            KeyError: If no observer can provide the requested risk factor.
        """
        for observer in self.observers:
            try:
                return observer.observe_risk_factor(identifier, time, state, attributes)
            except KeyError:
                continue
        raise KeyError(
            f"Risk factor '{identifier}' not found in any observer in composite '{self.name}'"
        )

    def _get_event_data(
        self,
        identifier: str,
        event_type: EventType,
        time: ActusDateTime,
        state: ContractState | None,
        attributes: ContractAttributes | None,
    ) -> Any:
        """Try each observer in order for event data.

        Args:
            identifier: Event data identifier.
            event_type: Type of event.
            time: Time at which to observe.
            state: Current contract state.
            attributes: Contract attributes.

        Returns:
            Event data from first matching observer.

        Raises:
            KeyError: If no observer can provide the requested event data.
        """
        for observer in self.observers:
            try:
                return observer.observe_event(identifier, event_type, time, state, attributes)
            except KeyError:
                continue
        raise KeyError(
            f"Event data '{identifier}' not found in any observer in composite '{self.name}'"
        )


class JaxRiskFactorObserver:
    """Fully JAX-compatible risk factor observer.

    This observer is designed for use with jax.jit and jax.grad. It uses
    integer indices instead of string identifiers and stores all data in
    JAX arrays.

    Key features:
    - Pure functions (no side effects)
    - JIT-compilable
    - Differentiable with jax.grad
    - Vectorized with jax.vmap
    - No Python control flow in hot paths

    Example:
        >>> # Create observer with 3 risk factors
        >>> risk_factors = jnp.array([1.18, 0.05, 100000.0])
        >>> observer = JaxRiskFactorObserver(risk_factors)
        >>>
        >>> # Observe risk factor at index 0 (e.g., FX rate)
        >>> fx_rate = observer.get(0)  # Returns 1.18
        >>>
        >>> # Use with jax.grad for sensitivities
        >>> def contract_value(risk_factors):
        ...     obs = JaxRiskFactorObserver(risk_factors)
        ...     fx = obs.get(0)
        ...     rate = obs.get(1)
        ...     notional = obs.get(2)
        ...     return notional * rate * fx
        >>>
        >>> # Compute gradient (sensitivities)
        >>> sensitivities = jax.grad(contract_value)(risk_factors)
        >>> # sensitivities[0] = d(value)/d(fx_rate)
        >>> # sensitivities[1] = d(value)/d(rate)
        >>> # sensitivities[2] = d(value)/d(notional)

    Note:
        This observer does not implement the RiskFactorObserver protocol
        because the protocol uses string identifiers which are not JAX-compatible.
        Instead, it provides a simpler API with integer indices.

    References:
        ACTUS v1.1 Section 2.9 - Risk Factor Observer
    """

    def __init__(
        self,
        risk_factors: jnp.ndarray,
        default_value: float | jnp.ndarray = 0.0,
    ):
        """Initialize JAX-compatible risk factor observer.

        Args:
            risk_factors: JAX array of risk factor values, indexed by integer
            default_value: Default value to return for out-of-bounds indices

        Example:
            >>> # Create observer with FX rate, interest rate, notional
            >>> risk_factors = jnp.array([1.18, 0.05, 100000.0])
            >>> observer = JaxRiskFactorObserver(risk_factors)
        """
        self.risk_factors = jnp.asarray(risk_factors, dtype=jnp.float32)
        self.default_value = jnp.array(default_value, dtype=jnp.float32)
        self.size = self.risk_factors.shape[0]

    def get(self, index: int) -> jnp.ndarray:
        """Get risk factor value at the given index.

        This method is JIT-compilable and differentiable.

        Args:
            index: Integer index of the risk factor

        Returns:
            Risk factor value as JAX array

        Example:
            >>> observer = JaxRiskFactorObserver(jnp.array([1.18, 0.05]))
            >>> fx_rate = observer.get(0)  # Returns 1.18
            >>> rate = observer.get(1)     # Returns 0.05

        Note:
            Uses safe indexing with bounds checking via JAX operations.
            Out-of-bounds indices return the default value.
        """
        # Safe indexing: return default_value if index is out of bounds
        # This is JAX-compatible (no Python if/else)
        valid = (index >= 0) & (index < self.size)
        return jnp.where(valid, self.risk_factors[index], self.default_value)

    def get_batch(self, indices: jnp.ndarray) -> jnp.ndarray:
        """Get multiple risk factors at once (vectorized).

        This is useful for batch operations and is vmappable.

        Args:
            indices: Array of integer indices

        Returns:
            Array of risk factor values

        Example:
            >>> observer = JaxRiskFactorObserver(jnp.array([1.18, 0.05, 100000.0]))
            >>> indices = jnp.array([0, 2])  # Get FX rate and notional
            >>> values = observer.get_batch(indices)
            >>> # Returns jnp.array([1.18, 100000.0])
        """
        # Vectorized safe indexing
        valid = (indices >= 0) & (indices < self.size)
        return jnp.where(valid, self.risk_factors[indices], self.default_value)

    def update(self, index: int, value: float) -> JaxRiskFactorObserver:
        """Create new observer with updated risk factor value.

        This is a pure function - it returns a new observer without
        modifying the original.

        Args:
            index: Index of risk factor to update
            value: New value

        Returns:
            New JaxRiskFactorObserver with updated value

        Example:
            >>> observer = JaxRiskFactorObserver(jnp.array([1.18, 0.05]))
            >>> new_observer = observer.update(0, 1.20)  # Update FX rate
            >>> new_observer.get(0)  # Returns 1.20
            >>> observer.get(0)      # Still returns 1.18 (original unchanged)
        """
        new_risk_factors = self.risk_factors.at[index].set(value)
        return JaxRiskFactorObserver(new_risk_factors, self.default_value)

    def update_batch(self, indices: jnp.ndarray, values: jnp.ndarray) -> JaxRiskFactorObserver:
        """Create new observer with multiple updated risk factors.

        Args:
            indices: Array of indices to update
            values: Array of new values

        Returns:
            New JaxRiskFactorObserver with updated values

        Example:
            >>> observer = JaxRiskFactorObserver(jnp.array([1.18, 0.05, 100000.0]))
            >>> new_observer = observer.update_batch(
            ...     jnp.array([0, 1]),
            ...     jnp.array([1.20, 0.06])
            ... )
        """
        new_risk_factors = self.risk_factors.at[indices].set(values)
        return JaxRiskFactorObserver(new_risk_factors, self.default_value)

    def to_array(self) -> jnp.ndarray:
        """Get all risk factors as a JAX array.

        Returns:
            Array of all risk factor values

        Example:
            >>> observer = JaxRiskFactorObserver(jnp.array([1.18, 0.05]))
            >>> observer.to_array()  # Returns jnp.array([1.18, 0.05])
        """
        return self.risk_factors

    @staticmethod
    def from_dict(
        mapping: dict[int, float], size: int | None = None, default_value: float = 0.0
    ) -> JaxRiskFactorObserver:
        """Create observer from a dictionary mapping indices to values.

        This is a convenience method for initialization. The observer itself
        remains fully JAX-compatible.

        Args:
            mapping: Dictionary mapping integer indices to float values
            size: Size of risk factor array (if None, uses max index + 1)
            default_value: Default value for unspecified indices

        Returns:
            New JaxRiskFactorObserver

        Example:
            >>> observer = JaxRiskFactorObserver.from_dict({
            ...     0: 1.18,    # FX rate
            ...     1: 0.05,    # Interest rate
            ...     2: 100000.0 # Notional
            ... })
        """
        if not mapping:
            risk_factors = jnp.array([], dtype=jnp.float32)
        else:
            max_index = max(mapping.keys())
            array_size = size if size is not None else max_index + 1
            risk_factors = jnp.full(array_size, default_value, dtype=jnp.float32)
            for idx, value in mapping.items():
                risk_factors = risk_factors.at[idx].set(value)

        return JaxRiskFactorObserver(risk_factors, default_value)
