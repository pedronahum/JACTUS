"""State transition functions for ACTUS contracts.

This module implements the State Transition Function (STF) framework, which defines
how contract states evolve through events.

The STF has the signature: STF(e, S_pre, M, t, o_rf) -> S_post
where:
- e: Event type
- S_pre: Pre-event contract state
- M: Contract attributes (terms)
- t: Event time
- o_rf: Risk factor observer

References:
    ACTUS v1.1 Section 2.8 - State Transition Functions
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Protocol, runtime_checkable

import jax.numpy as jnp

from jactus.core import ActusDateTime, ContractState
from jactus.core.types import EventType, FeeBasis
from jactus.utilities import year_fraction

if TYPE_CHECKING:
    from jactus.core import ContractAttributes


@runtime_checkable
class StateTransitionFunction(Protocol):
    """Protocol for state transition functions.

    The state transition function signature is:
        STF(e, S_pre, M, t, o_rf) -> S_post

    All implementations must be JAX-compatible (pure functions, JIT-compilable).
    """

    def __call__(
        self,
        event_type: EventType,
        state_pre: ContractState,
        attributes: ContractAttributes,
        time: ActusDateTime,
        risk_factor_observer: RiskFactorObserver,  # type: ignore # noqa: F821
    ) -> ContractState:
        """Transition contract state through an event.

        Args:
            event_type: Type of event triggering the transition
            state_pre: Pre-event contract state
            attributes: Contract attributes (terms)
            time: Event time
            risk_factor_observer: Observer for risk factors

        Returns:
            Post-event contract state
        """
        ...


class BaseStateTransitionFunction(ABC):
    """Base class for state transition functions with common logic.

    This class provides a framework for implementing state transitions with
    standard patterns like interest accrual, fee accrual, and status date updates.

    Subclasses must implement the abstract `transition_state()` method with
    contract-specific state transition logic.
    """

    def __init__(self, day_count_convention: DayCountConvention | None = None):  # type: ignore # noqa: F821
        """Initialize state transition function.

        Args:
            day_count_convention: Day count convention for time calculations
                (if None, uses contract's DCC from attributes)
        """
        self.day_count_convention = day_count_convention

    @abstractmethod
    def transition_state(
        self,
        event_type: EventType,
        state_pre: ContractState,
        attributes: ContractAttributes,
        time: ActusDateTime,
        risk_factor_observer: RiskFactorObserver,  # type: ignore # noqa: F821
    ) -> ContractState:
        """Implement contract-specific state transition logic.

        This method should be implemented by subclasses to define how the
        contract state transitions for specific event types.

        Args:
            event_type: Type of event triggering the transition
            state_pre: Pre-event contract state
            attributes: Contract attributes (terms)
            time: Event time
            risk_factor_observer: Observer for risk factors

        Returns:
            Post-event contract state

        Note:
            This method should typically:
            1. Copy the pre-event state
            2. Update state variables based on event type
            3. Return the new state

        Example:
            >>> def transition_state(self, event_type, state_pre, attributes, time, observer):
            ...     # Create post-event state by copying pre-event state
            ...     state_post = state_pre
            ...
            ...     # Update state based on event type
            ...     if event_type == EventType.IP:
            ...         # Interest payment: reset accrued interest
            ...         state_post = state_post.replace(ipac=jnp.array(0.0))
            ...
            ...     return state_post
        """
        ...

    def update_status_date(self, state: ContractState, new_date: ActusDateTime) -> ContractState:
        """Update contract status date.

        Args:
            state: Contract state to update
            new_date: New status date

        Returns:
            Updated state with new status date
        """
        return state.replace(sd=new_date)

    def accrue_interest(
        self,
        state: ContractState,
        attributes: ContractAttributes,
        from_date: ActusDateTime,
        to_date: ActusDateTime,
    ) -> jnp.ndarray:
        """Calculate interest accrued over a period.

        Accrual formula: IPAC += NT * IPNR * YearFraction(from, to, DCC)

        Args:
            state: Current contract state
            attributes: Contract attributes
            from_date: Start of accrual period
            to_date: End of accrual period

        Returns:
            Interest accrued as JAX array

        References:
            ACTUS v1.1 Section 3.4 - Interest Accrual
        """
        # Get day count convention
        dcc = self.day_count_convention or attributes.day_count_convention

        # Calculate year fraction
        yf = year_fraction(from_date, to_date, dcc)  # type: ignore[arg-type]

        # Calculate accrued interest: NT * IPNR * YF
        return state.nt * state.ipnr * jnp.array(yf, dtype=jnp.float32)

    def accrue_fees(
        self,
        state: ContractState,
        attributes: ContractAttributes,
        from_date: ActusDateTime,
        to_date: ActusDateTime,
        fee_rate: jnp.ndarray,
        fee_basis: FeeBasis,
    ) -> jnp.ndarray:
        """Calculate fees accrued over a period.

        Fee accrual depends on the fee basis:
        - FeeBasis.A (Absolute): FEAC += FER * YearFraction(from, to, DCC)
        - FeeBasis.N (Notional): FEAC += NT * FER * YearFraction(from, to, DCC)

        Args:
            state: Current contract state
            attributes: Contract attributes
            from_date: Start of accrual period
            to_date: End of accrual period
            fee_rate: Fee rate (FER)
            fee_basis: Fee basis (Absolute or Notional)

        Returns:
            Fees accrued as JAX array

        References:
            ACTUS v1.1 Section 3.5 - Fee Accrual
        """
        # Get day count convention
        dcc = self.day_count_convention or attributes.day_count_convention

        # Calculate year fraction
        yf = year_fraction(from_date, to_date, dcc)  # type: ignore[arg-type]

        # Calculate accrued fees based on basis
        if fee_basis == FeeBasis.A:
            # Absolute basis: FER * YF
            accrued = fee_rate * jnp.array(yf, dtype=jnp.float32)
        else:  # FeeBasis.N
            # Notional basis: NT * FER * YF
            accrued = state.nt * fee_rate * jnp.array(yf, dtype=jnp.float32)

        return accrued

    def __call__(
        self,
        event_type: EventType,
        state_pre: ContractState,
        attributes: ContractAttributes,
        time: ActusDateTime,
        risk_factor_observer: RiskFactorObserver,  # type: ignore # noqa: F821
    ) -> ContractState:
        """Execute complete state transition pipeline.

        This method:
        1. Calls the abstract `transition_state()` method
        2. Updates the status date to the event time
        3. Returns the post-event state

        Args:
            event_type: Type of event triggering the transition
            state_pre: Pre-event contract state
            attributes: Contract attributes (terms)
            time: Event time
            risk_factor_observer: Observer for risk factors

        Returns:
            Post-event contract state

        References:
            ACTUS v1.1 Section 2.8
        """
        # Execute contract-specific state transition
        state_post = self.transition_state(
            event_type, state_pre, attributes, time, risk_factor_observer
        )

        # Update status date to event time
        return self.update_status_date(state_post, time)


def create_state_pre(
    tmd: ActusDateTime,
    sd: ActusDateTime,
    nt: float,
    ipnr: float,
    ipac: float = 0.0,
    feac: float = 0.0,
    nsc: float = 1.0,
    isc: float = 1.0,
) -> ContractState:
    """Create a pre-event contract state with default values.

    This is a convenience function for creating initial contract states
    with common default values.

    Args:
        tmd: Time of maturity date
        sd: Status date
        nt: Notional principal
        ipnr: Nominal interest rate
        ipac: Accrued interest (default: 0.0)
        feac: Accrued fees (default: 0.0)
        nsc: Notional scaling multiplier (default: 1.0)
        isc: Interest scaling multiplier (default: 1.0)

    Returns:
        Initial contract state

    Example:
        >>> state = create_state_pre(
        ...     tmd=ActusDateTime(2029, 1, 15, 0, 0, 0),
        ...     sd=ActusDateTime(2024, 1, 15, 0, 0, 0),
        ...     nt=100000.0,
        ...     ipnr=0.05,
        ... )
    """
    return ContractState(
        tmd=tmd,
        sd=sd,
        nt=jnp.array(nt, dtype=jnp.float32),
        ipnr=jnp.array(ipnr, dtype=jnp.float32),
        ipac=jnp.array(ipac, dtype=jnp.float32),
        feac=jnp.array(feac, dtype=jnp.float32),
        nsc=jnp.array(nsc, dtype=jnp.float32),
        isc=jnp.array(isc, dtype=jnp.float32),
    )


def validate_state_transition(
    state_pre: ContractState,
    state_post: ContractState,
    event_type: EventType,  # noqa: ARG001
) -> bool:
    """Validate that a state transition is consistent.

    Performs basic sanity checks on state transitions:
    - Status date should not go backwards
    - Notional should not become negative (for most events)
    - Scaling factors should remain positive

    Args:
        state_pre: Pre-event state
        state_post: Post-event state
        event_type: Event type that caused the transition

    Returns:
        True if transition is valid, False otherwise

    Example:
        >>> is_valid = validate_state_transition(state_pre, state_post, EventType.IP)
        >>> if not is_valid:
        ...     raise ValueError("Invalid state transition")
    """
    # Status date should not go backwards
    if state_post.sd < state_pre.sd:
        return False

    # Notional should not be negative (except for some edge cases)
    if state_post.nt < 0:
        return False

    # Scaling factors should be positive
    return not (state_post.nsc <= 0 or state_post.isc <= 0)
