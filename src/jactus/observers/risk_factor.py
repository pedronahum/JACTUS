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

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import jax.numpy as jnp

if TYPE_CHECKING:
    from jactus.core import ActusDateTime, ContractAttributes, ContractState


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
        event_type: EventType,  # type: ignore # noqa: F821
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
        event_type: EventType,  # type: ignore # noqa: F821
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
        event_type: EventType,  # type: ignore # noqa: F821
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
        event_type: EventType,  # type: ignore # noqa: F821, ARG002
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
        event_type: EventType,  # type: ignore # noqa: F821, ARG002
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
