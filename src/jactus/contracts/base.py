"""Base contract class for all ACTUS contracts.

This module implements the abstract base class that all ACTUS contract types
inherit from. It provides the core simulation engine and common functionality.

References:
    ACTUS v1.1 Section 3 - Contract Types
    ACTUS v1.1 Section 4 - Event Schedules
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import flax.nnx as nnx
import jax.numpy as jnp

from jactus.core import (
    ActusDateTime,
    ContractAttributes,
    ContractEvent,
    ContractState,
    EventSchedule,
)
from jactus.functions import PayoffFunction, StateTransitionFunction
from jactus.observers import ChildContractObserver, RiskFactorObserver


@dataclass
class SimulationHistory:
    """Results from contract simulation.

    Contains the complete history of events and states from a contract
    simulation run.

    Attributes:
        events: List of all events (scheduled + observed)
        states: List of states (one per event, plus initial)
        initial_state: Contract state before first event
        final_state: Contract state after last event

    Example:
        >>> history = contract.simulate(observers)
        >>> print(f"Generated {len(history.events)} events")
        >>> print(f"Final notional: {history.final_state.nt}")
    """

    events: list[ContractEvent]
    states: list[ContractState]
    initial_state: ContractState
    final_state: ContractState

    def get_cashflows(self) -> list[tuple[ActusDateTime, jnp.ndarray, str]]:
        """Extract cashflow timeline from events.

        Returns:
            List of (time, payoff, currency) tuples

        Example:
            >>> cashflows = history.get_cashflows()
            >>> for time, amount, currency in cashflows:
            ...     print(f"{time.to_iso()}: {amount} {currency}")
        """
        return [(e.event_time, e.payoff, e.currency) for e in self.events]

    def filter_events(
        self, start: ActusDateTime | None = None, end: ActusDateTime | None = None
    ) -> list[ContractEvent]:
        """Filter events by time range.

        Args:
            start: Optional start time (inclusive)
            end: Optional end time (inclusive)

        Returns:
            List of events in the specified range

        Example:
            >>> year_events = history.filter_events(
            ...     start=ActusDateTime(2024, 1, 1, 0, 0, 0),
            ...     end=ActusDateTime(2024, 12, 31, 23, 59, 59)
            ... )
        """
        filtered = self.events
        if start is not None:
            filtered = [e for e in filtered if e.event_time >= start]
        if end is not None:
            filtered = [e for e in filtered if e.event_time <= end]
        return filtered


class BaseContract(nnx.Module, ABC):
    """Abstract base class for all ACTUS contracts.

    This class provides the core simulation engine and common functionality
    that all contract types share. Subclasses must implement the abstract
    methods to define contract-specific behavior.

    The class extends flax.nnx.Module to enable:
    - JIT compilation of contract simulation
    - Automatic differentiation for sensitivity analysis
    - Vectorization over multiple contracts or scenarios
    - Pytree compatibility for JAX transformations

    Attributes:
        attributes: Contract attributes (terms and conditions)
        risk_factor_observer: Observer for market risk factors
        child_contract_observer: Optional observer for child contracts
        _event_cache: Cached event schedule (None until first computation)

    Example:
        >>> class MyContract(BaseContract):
        ...     def generate_event_schedule(self):
        ...         # Generate contract-specific events
        ...         pass
        ...     # ... implement other abstract methods
        >>> contract = MyContract(attributes, risk_observer)
        >>> history = contract.simulate()
        >>> npv = contract.calculate_npv(discount_rate=0.05)

    References:
        ACTUS v1.1 Section 3 - Contract Types
        ACTUS v1.1 Section 4 - Algorithm
    """

    def __init__(
        self,
        attributes: ContractAttributes,
        risk_factor_observer: RiskFactorObserver,
        child_contract_observer: ChildContractObserver | None = None,
        *,
        rngs: nnx.Rngs | None = None,
    ):
        """Initialize base contract.

        Args:
            attributes: Contract attributes (terms and conditions)
            risk_factor_observer: Observer for accessing market risk factors
            child_contract_observer: Optional observer for child contracts
            rngs: Optional Flax RNG state for stochastic contracts

        Example:
            >>> from jactus.observers import ConstantRiskFactorObserver
            >>> attrs = ContractAttributes(...)
            >>> risk_obs = ConstantRiskFactorObserver(1.0)
            >>> contract = MyContract(attrs, risk_obs)
        """
        super().__init__()
        self.attributes = attributes
        self.risk_factor_observer = risk_factor_observer
        self.child_contract_observer = child_contract_observer
        self.rngs = rngs if rngs is not None else nnx.Rngs(0)
        self._event_cache: EventSchedule | None = None

    # ========================================================================
    # Abstract methods - must be implemented by subclasses
    # ========================================================================

    @abstractmethod
    def generate_event_schedule(self) -> EventSchedule:
        """Generate the scheduled events for this contract.

        This method must be implemented by each contract type to generate
        its specific event schedule according to ACTUS rules.

        Returns:
            EventSchedule containing all scheduled events

        Example:
            >>> def generate_event_schedule(self):
            ...     events = []
            ...     # Add IED event
            ...     events.append(ContractEvent(
            ...         event_type=EventType.IED,
            ...         event_time=self.attributes.initial_exchange_date,
            ...         ...
            ...     ))
            ...     # Add other events...
            ...     return EventSchedule(events)

        References:
            ACTUS v1.1 Section 4.1 - Event Schedule Generation
        """
        raise NotImplementedError

    @abstractmethod
    def initialize_state(self) -> ContractState:
        """Initialize contract state before first event.

        Creates the initial state based on contract attributes. This state
        is used as the starting point for simulation.

        Returns:
            Initial ContractState

        Example:
            >>> def initialize_state(self):
            ...     return ContractState(
            ...         sd=self.attributes.status_date,
            ...         tmd=self.attributes.maturity_date,
            ...         nt=jnp.array(self.attributes.notional_principal),
            ...         ipnr=jnp.array(self.attributes.nominal_interest_rate),
            ...         ...
            ...     )

        References:
            ACTUS v1.1 Section 4.2 - State Initialization
        """
        raise NotImplementedError

    @abstractmethod
    def get_payoff_function(self, event_type: Any) -> PayoffFunction:
        """Get payoff function for a specific event type.

        Returns the appropriate payoff function (POF) for calculating
        the cashflow generated by the given event type.

        Args:
            event_type: Event type (e.g., EventType.IP, EventType.PR)

        Returns:
            PayoffFunction for the event type

        Example:
            >>> def get_payoff_function(self, event_type):
            ...     if event_type == EventType.IP:
            ...         return InterestPaymentPayoff()
            ...     elif event_type == EventType.PR:
            ...         return PrincipalRedemptionPayoff()
            ...     else:
            ...         return ZeroPayoff()

        References:
            ACTUS v1.1 Section 2.7 - Payoff Functions
        """
        raise NotImplementedError

    @abstractmethod
    def get_state_transition_function(self, event_type: Any) -> StateTransitionFunction:
        """Get state transition function for a specific event type.

        Returns the appropriate state transition function (STF) for
        updating contract state when the given event occurs.

        Args:
            event_type: Event type (e.g., EventType.IP, EventType.PR)

        Returns:
            StateTransitionFunction for the event type

        Example:
            >>> def get_state_transition_function(self, event_type):
            ...     if event_type == EventType.IP:
            ...         return InterestPaymentSTF()
            ...     elif event_type == EventType.PR:
            ...         return PrincipalRedemptionSTF()
            ...     else:
            ...         return IdentitySTF()

        References:
            ACTUS v1.1 Section 2.8 - State Transition Functions
        """
        raise NotImplementedError

    # ========================================================================
    # Concrete methods - common to all contracts
    # ========================================================================

    def get_lifetime(self) -> tuple[ActusDateTime, ActusDateTime]:
        """Get contract lifetime (start and end dates).

        Returns:
            Tuple of (start_date, end_date)

        Example:
            >>> start, end = contract.get_lifetime()
            >>> print(f"Contract runs from {start.to_iso()} to {end.to_iso()}")
        """
        start = self.attributes.status_date
        # End date logic - will be refined in D2.6
        end = self.attributes.maturity_date or start
        return start, end

    def is_maturity_contract(self) -> bool:
        """Check if contract has a defined maturity date.

        Returns:
            True if contract has maturity date, False otherwise

        Example:
            >>> if contract.is_maturity_contract():
            ...     print("Contract matures at", contract.attributes.maturity_date)
        """
        return self.attributes.maturity_date is not None

    def get_events(self, force_regenerate: bool = False) -> EventSchedule:
        """Get event schedule with caching.

        Generates and caches the event schedule on first call. Subsequent
        calls return the cached schedule unless force_regenerate=True.

        Args:
            force_regenerate: If True, regenerate schedule even if cached

        Returns:
            EventSchedule containing all scheduled events

        Example:
            >>> events = contract.get_events()
            >>> print(f"Contract has {len(events)} events")
            >>> # Regenerate if attributes changed
            >>> events = contract.get_events(force_regenerate=True)
        """
        if self._event_cache is None or force_regenerate:
            self._event_cache = self.generate_event_schedule()
        return self._event_cache

    def get_events_in_range(
        self,
        start: ActusDateTime | None = None,
        end: ActusDateTime | None = None,
    ) -> list[ContractEvent]:
        """Get events within a time range.

        Args:
            start: Optional start time (inclusive)
            end: Optional end time (inclusive)

        Returns:
            List of events in the specified range

        Example:
            >>> # Get all events in 2024
            >>> events_2024 = contract.get_events_in_range(
            ...     start=ActusDateTime(2024, 1, 1, 0, 0, 0),
            ...     end=ActusDateTime(2024, 12, 31, 23, 59, 59)
            ... )
        """
        schedule = self.get_events()
        filtered = list(schedule.events)
        if start is not None:
            filtered = [e for e in filtered if e.event_time >= start]
        if end is not None:
            filtered = [e for e in filtered if e.event_time <= end]
        return filtered

    def simulate(
        self,
        risk_factor_observer: RiskFactorObserver | None = None,
        child_contract_observer: ChildContractObserver | None = None,  # noqa: ARG002
    ) -> SimulationHistory:
        """Simulate contract through all events.

        Executes the full ACTUS algorithm:
        1. Initialize state
        2. For each event:
           a. Apply state transition function (STF)
           b. Calculate payoff (POF)
           c. Store event with states

        Args:
            risk_factor_observer: Optional override for risk factor observer
            child_contract_observer: Optional override for child contract observer

        Returns:
            SimulationHistory with events and states

        Example:
            >>> history = contract.simulate()
            >>> for event in history.events:
            ...     print(f"{event.event_time}: {event.payoff}")

        References:
            ACTUS v1.1 Section 4 - Algorithm
        """
        # Use provided observers or fall back to instance observers
        risk_obs = risk_factor_observer or self.risk_factor_observer
        # child_contract_observer available for future use with composite contracts

        # Initialize
        state = self.initialize_state()
        initial_state = state
        events_with_states = []

        # Get scheduled events
        schedule = self.get_events()

        # Process each event
        for event in schedule.events:
            # Get functions for this event type
            stf = self.get_state_transition_function(event.event_type)
            pof = self.get_payoff_function(event.event_type)

            # For CS (Calculate/Shift) BDC conventions, use the original
            # unadjusted date for calculations (year fraction, accrual).
            # For SC (Shift/Calculate) or NULL, calculation_time is None
            # and we use event_time as before.
            calc_time = event.calculation_time or event.event_time

            # Calculate payoff BEFORE state transition (using pre-event state)
            payoff = pof(
                event_type=event.event_type,
                state=state,
                attributes=self.attributes,
                time=calc_time,
                risk_factor_observer=risk_obs,
            )

            # Apply state transition AFTER payoff calculation
            state_post = stf(
                event_type=event.event_type,
                state_pre=state,
                attributes=self.attributes,
                time=calc_time,
                risk_factor_observer=risk_obs,
            )

            # Create event with states and payoff
            processed_event = ContractEvent(
                event_type=event.event_type,
                event_time=event.event_time,
                payoff=payoff,
                currency=self.attributes.currency or "XXX",
                state_pre=state,
                state_post=state_post,
                sequence=event.sequence,
            )

            events_with_states.append(processed_event)
            state = state_post

        return SimulationHistory(
            events=events_with_states,
            states=[e.state_post for e in events_with_states if e.state_post is not None],
            initial_state=initial_state,
            final_state=state,
        )

    def get_cashflows(
        self,
        risk_factor_observer: RiskFactorObserver | None = None,
        child_contract_observer: ChildContractObserver | None = None,
    ) -> list[tuple[ActusDateTime, jnp.ndarray, str]]:
        """Get cashflow timeline from contract.

        Convenience method that simulates and extracts cashflows.

        Args:
            risk_factor_observer: Optional override for risk factor observer
            child_contract_observer: Optional override for child contract observer

        Returns:
            List of (time, payoff, currency) tuples

        Example:
            >>> cashflows = contract.get_cashflows()
            >>> total = sum(payoff for _, payoff, _ in cashflows)
            >>> print(f"Total cashflows: {total}")
        """
        history = self.simulate(risk_factor_observer, child_contract_observer)
        return history.get_cashflows()

    def validate(self) -> dict[str, list[str]]:
        """Validate contract attributes.

        Checks contract attributes for consistency and completeness.
        Returns any validation errors or warnings.

        Returns:
            Dictionary with 'errors' and 'warnings' lists

        Example:
            >>> result = contract.validate()
            >>> if result['errors']:
            ...     print("Validation failed:", result['errors'])
            >>> if result['warnings']:
            ...     print("Warnings:", result['warnings'])

        Note:
            Base implementation performs basic checks. Subclasses should
            override to add contract-specific validation.
        """
        errors = []
        warnings = []

        # Check required attributes
        if not self.attributes.contract_id:
            errors.append("contract_id is required")

        if not self.attributes.status_date:
            errors.append("status_date is required")

        # Check notional principal
        if (
            self.attributes.notional_principal is not None
            and self.attributes.notional_principal <= 0
        ):
            warnings.append("notional_principal should be positive")

        # Check interest rate
        if (
            self.attributes.nominal_interest_rate is not None
            and abs(self.attributes.nominal_interest_rate) > 1.0
        ):
            warnings.append("nominal_interest_rate seems unusually high (>100%)")

        return {"errors": errors, "warnings": warnings}


# ============================================================================
# Helper functions
# ============================================================================


def sort_events_by_sequence(events: list[ContractEvent]) -> list[ContractEvent]:
    """Sort events by time and sequence number.

    Events are sorted first by time, then by sequence number for events
    at the same time. This ensures deterministic event ordering.

    Args:
        events: List of events to sort

    Returns:
        Sorted list of events

    Example:
        >>> events = [event3, event1, event2]
        >>> sorted_events = sort_events_by_sequence(events)
        >>> assert sorted_events[0].event_time <= sorted_events[1].event_time

    Note:
        This is a pure function - it does not modify the input list.
    """
    return sorted(events, key=lambda e: (e.event_time, e.sequence))


def merge_scheduled_and_observed_events(
    scheduled: list[ContractEvent],
    observed: list[ContractEvent],
) -> list[ContractEvent]:
    """Merge scheduled and observed events.

    Combines scheduled events (from generate_event_schedule) with
    observed events (from child contract or risk factor observers).
    Removes duplicates and sorts by time and sequence.

    Args:
        scheduled: List of scheduled events
        observed: List of observed events

    Returns:
        Merged and sorted list of events

    Example:
        >>> all_events = merge_scheduled_and_observed_events(
        ...     scheduled_events,
        ...     observed_events
        ... )

    Note:
        If two events have the same time and type, only the first is kept.
        This prevents duplicate event processing.
    """
    # Combine lists
    all_events = scheduled + observed

    # Remove duplicates (same time and event_type)
    seen = set()
    unique_events = []
    for event in all_events:
        key = (event.event_time, event.event_type)
        if key not in seen:
            seen.add(key)
            unique_events.append(event)

    # Sort by time and sequence
    return sort_events_by_sequence(unique_events)
