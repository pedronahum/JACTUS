"""Behavioral risk factor observers for ACTUS contracts.

This module implements the Behavioral Risk Factor Observer framework, which
provides state-dependent risk modeling capabilities. Unlike market risk factor
observers that return values based solely on identifiers and time, behavioral
observers are aware of the contract's internal state (notional, interest rate,
age, performance status, etc.) and can dynamically inject events into the
simulation timeline.

Key concepts:
- **CalloutEvent**: An event that a behavioral model requests be added to the
  simulation schedule. When the simulation reaches that time, it calls back to
  the behavioral observer with the current contract state.
- **BehaviorRiskFactorObserver**: Protocol extending RiskFactorObserver with
  a ``contract_start`` method that returns callout events.
- **BaseBehaviorRiskFactorObserver**: Abstract base class providing the
  framework for implementing concrete behavioral models.

This architecture mirrors the ACTUS risk service's separation of market risk
(pure time-series observation) from behavioral risk (state-dependent models
like prepayment and deposit transaction behavior).

References:
    ACTUS Risk Service v2.0 - BehaviorRiskModelProvider interface
    ACTUS Technical Specification v1.1, Section 2.9 - Risk Factor Observer
"""

from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import jax.numpy as jnp

from jactus.observers.risk_factor import BaseRiskFactorObserver

if TYPE_CHECKING:
    from jactus.core import ActusDateTime, ContractAttributes, ContractState
    from jactus.core.types import EventType


@dataclass(frozen=True)
class CalloutEvent:
    """An event requested by a behavioral risk model for injection into the simulation schedule.

    Behavioral models return these from ``contract_start()`` to register
    observation times. When the simulation engine reaches the specified time,
    it evaluates the behavioral observer with the current contract state.

    Attributes:
        model_id: Identifier of the behavioral model requesting this event
            (e.g., ``"ppm01"`` for a prepayment model).
        time: The time at which this observation event should occur.
        callout_type: Type code indicating the nature of the callout. Common
            types include:
            - ``"MRD"`` (Multiplicative Reduction Delta) for prepayment models
            - ``"AFD"`` (Absolute Funded Delta) for deposit transaction models
        metadata: Optional additional data the model needs at callout time
            (e.g., reference rate identifier, surface parameters).

    Example:
        >>> from jactus.core import ActusDateTime
        >>> event = CalloutEvent(
        ...     model_id="prepayment-model-01",
        ...     time=ActusDateTime(2025, 6, 1),
        ...     callout_type="MRD",
        ... )
    """

    model_id: str
    time: ActusDateTime
    callout_type: str
    metadata: dict[str, Any] | None = None


@runtime_checkable
class BehaviorRiskFactorObserver(Protocol):
    """Protocol for behavioral risk factor observers.

    Extends the standard risk factor observer concept with state-dependent
    observation and dynamic event injection. Behavioral observers:

    1. Are called at the start of contract simulation via ``contract_start()``
       to register observation times (callout events).
    2. At those times, are called via ``observe_risk_factor()`` with the full
       contract state, enabling state-dependent calculations.

    This mirrors the ACTUS risk service's ``BehaviorRiskModelProvider``
    interface, which separates behavioral models from pure market data.

    All implementations should be JAX-compatible where possible.
    """

    def observe_risk_factor(
        self,
        identifier: str,
        time: ActusDateTime,
        state: ContractState | None = None,
        attributes: ContractAttributes | None = None,
    ) -> jnp.ndarray:
        """Observe a behavioral risk factor at a specific time.

        Unlike market risk observers, behavioral observers typically use
        the ``state`` and ``attributes`` parameters to compute their output.
        For example, a prepayment model computes the spread between the
        contract's nominal interest rate and the current market rate.

        Args:
            identifier: Risk factor identifier (e.g., model ID).
            time: Time at which to observe.
            state: Current contract state (used for state-dependent factors).
            attributes: Contract attributes (used for contract-dependent factors).

        Returns:
            Risk factor value as JAX array.
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
        """Observe event-related behavioral data.

        Args:
            identifier: Event data identifier.
            event_type: Type of event.
            time: Time at which to observe.
            state: Current contract state.
            attributes: Contract attributes.

        Returns:
            Event-related data.
        """
        ...

    def contract_start(
        self,
        attributes: ContractAttributes,
    ) -> list[CalloutEvent]:
        """Called at the start of contract simulation.

        The behavioral model inspects the contract attributes to determine
        when it needs to be evaluated during the simulation. It returns a
        list of ``CalloutEvent`` objects that the simulation engine merges
        into the event schedule.

        For example, a prepayment model might return semi-annual observation
        events over the life of the contract.

        Args:
            attributes: Contract attributes (terms and conditions).

        Returns:
            List of callout events to inject into the simulation schedule.
            May be empty if the model does not need scheduled observations.

        Example:
            >>> events = observer.contract_start(attributes)
            >>> for e in events:
            ...     print(f"{e.model_id} @ {e.time}: {e.callout_type}")
        """
        ...


class BaseBehaviorRiskFactorObserver(BaseRiskFactorObserver):
    """Abstract base class for behavioral risk factor observers.

    Provides the framework for implementing state-dependent risk models
    that can inject callout events into the simulation schedule. Subclasses
    must implement:

    - ``_get_risk_factor()``: State-aware risk factor computation
    - ``_get_event_data()``: Event data retrieval
    - ``contract_start()``: Callout event generation

    Example:
        >>> class MyBehavioralModel(BaseBehaviorRiskFactorObserver):
        ...     def _get_risk_factor(self, identifier, time, state, attributes):
        ...         # Use state.ipnr (contract rate) in computation
        ...         spread = float(state.ipnr) - 0.03
        ...         return jnp.array(max(spread, 0.0) * 0.1)
        ...
        ...     def _get_event_data(self, identifier, event_type, time, state, attributes):
        ...         raise KeyError("No event data")
        ...
        ...     def contract_start(self, attributes):
        ...         return [CalloutEvent("my-model", attributes.status_date, "MRD")]
    """

    def __init__(self, name: str | None = None):
        """Initialize behavioral risk factor observer.

        Args:
            name: Optional name for this observer (for debugging).
        """
        super().__init__(name)

    @abstractmethod
    def contract_start(
        self,
        attributes: ContractAttributes,
    ) -> list[CalloutEvent]:
        """Called at the start of contract simulation.

        Must be implemented by subclasses to return callout events that
        should be injected into the simulation schedule.

        Args:
            attributes: Contract attributes.

        Returns:
            List of CalloutEvent objects.
        """
        ...
