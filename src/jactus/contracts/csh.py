"""Cash (CSH) contract implementation.

This module implements the CSH (Cash) contract type, which represents a simple
cash position. It is the simplest ACTUS contract with only Analysis Date (AD)
events and minimal state.

ACTUS Reference:
    ACTUS v1.1 Section 7.8 - CSH: Cash

Example:
    >>> from jactus.contracts.csh import CashContract
    >>> from jactus.core import ContractAttributes, ContractType, ContractRole
    >>> from jactus.observers import ConstantRiskFactorObserver
    >>>
    >>> attrs = ContractAttributes(
    ...     contract_id="CASH-001",
    ...     contract_type=ContractType.CSH,
    ...     contract_role=ContractRole.RPA,
    ...     status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
    ...     currency="USD",
    ...     notional_principal=10000.0
    ... )
    >>>
    >>> rf_obs = ConstantRiskFactorObserver(constant_value=0.0)
    >>> contract = CashContract(attributes=attrs, risk_factor_observer=rf_obs)
    >>> result = contract.simulate()
"""

from typing import Any

import flax.nnx as nnx
import jax.numpy as jnp

from jactus.contracts.base import BaseContract
from jactus.core import (
    ActusDateTime,
    ContractAttributes,
    ContractEvent,
    ContractState,
    ContractType,
    EventSchedule,
    EventType,
)
from jactus.functions import BasePayoffFunction, BaseStateTransitionFunction
from jactus.observers import ChildContractObserver, RiskFactorObserver


class CashPayoffFunction(BasePayoffFunction):
    """Payoff function for Cash (CSH) contracts.

    CSH contracts have no actual cashflows - they only represent a position.
    All payoffs return 0.0.

    ACTUS Reference:
        ACTUS v1.1 Section 7.8 - CSH Payoff Functions

    Example:
        >>> pof = CashPayoffFunction(contract_role=ContractRole.RPA, currency="USD")
        >>> state = ContractState(...)
        >>> payoff = pof(EventType.AD, state, attrs, time, rf_obs)
        >>> print(payoff)  # 0.0
    """

    def calculate_payoff(
        self,
        event_type: Any,
        state: ContractState,
        attributes: ContractAttributes,
        time: ActusDateTime,
        risk_factor_observer: RiskFactorObserver,
    ) -> jnp.ndarray:
        """Calculate payoff for CSH events.

        CSH contracts have no cashflows - they only represent a position value.
        All events return 0.0 payoff.

        Args:
            event_type: Type of event (only AD for CSH)
            state: Current contract state
            attributes: Contract attributes
            time: Event time
            risk_factor_observer: Observer for market data

        Returns:
            Always returns 0.0 (no cashflow)

        ACTUS Reference:
            POF_AD_CSH = 0.0
        """
        # CSH has no actual cashflows
        return jnp.array(0.0, dtype=jnp.float32)


class CashStateTransitionFunction(BaseStateTransitionFunction):
    """State transition function for Cash (CSH) contracts.

    CSH contracts have minimal state transitions - only the status date
    is updated. Notional remains constant throughout the contract life.

    ACTUS Reference:
        ACTUS v1.1 Section 7.8 - CSH State Transition Functions

    Example:
        >>> stf = CashStateTransitionFunction()
        >>> state_post = stf(EventType.AD, state_pre, attrs, time, rf_obs)
        >>> assert state_post.sd == time
        >>> assert state_post.nt == state_pre.nt  # Notional unchanged
    """

    def transition_state(
        self,
        event_type: Any,
        state_pre: ContractState,
        attributes: ContractAttributes,
        time: ActusDateTime,
        risk_factor_observer: RiskFactorObserver,
    ) -> ContractState:
        """Transition CSH contract state.

        CSH state transitions are minimal - only the status date is updated.
        The notional (nt) remains constant.

        Args:
            event_type: Type of event (only AD for CSH)
            state_pre: State before event
            attributes: Contract attributes
            time: Event time
            risk_factor_observer: Observer for market data

        Returns:
            Updated contract state with new status date

        ACTUS Reference:
            STF_AD_CSH:
                sd_t = t
                nt_t = nt_t⁻ (unchanged)
        """
        # CSH: Only update status date, notional remains constant
        return ContractState(
            sd=time,
            tmd=state_pre.tmd,
            nt=state_pre.nt,
            ipnr=state_pre.ipnr,
            ipac=state_pre.ipac,
            feac=state_pre.feac,
            nsc=state_pre.nsc,
            isc=state_pre.isc,
        )


class CashContract(BaseContract):
    """Cash (CSH) contract implementation.

    Represents a simple cash position with no cashflows. This is the simplest
    ACTUS contract type, used for cash accounts, settlement amounts, or as
    initial positions in portfolios.

    Characteristics:
        - Only AD (Analysis Date) events
        - Minimal state: notional (nt) and status date (sd)
        - No cashflows (all payoffs = 0.0)
        - No interest accrual or fees

    ACTUS Reference:
        ACTUS v1.1 Section 7.8 - CSH: Cash

    Attributes:
        attributes: Contract attributes including notional, currency, role
        risk_factor_observer: Observer for market data (not used in CSH)
        child_contract_observer: Observer for child contracts (not used in CSH)
        rngs: Random number generators for JAX

    Example:
        >>> from jactus.contracts.csh import CashContract
        >>> from jactus.core import ContractAttributes, ContractType, ContractRole
        >>>
        >>> attrs = ContractAttributes(
        ...     contract_id="CASH-001",
        ...     contract_type=ContractType.CSH,
        ...     contract_role=ContractRole.RPA,
        ...     status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
        ...     currency="USD",
        ...     notional_principal=10000.0
        ... )
        >>>
        >>> contract = CashContract(
        ...     attributes=attrs,
        ...     risk_factor_observer=ConstantRiskFactorObserver(0.0)
        ... )
        >>>
        >>> # Simulate the contract
        >>> result = contract.simulate()
        >>> print(f"Events: {len(result.events)}")  # 1 (just AD)
        >>> print(f"Total cashflow: {result.total_cashflow()}")  # 0.0
    """

    def __init__(
        self,
        attributes: ContractAttributes,
        risk_factor_observer: RiskFactorObserver,
        child_contract_observer: ChildContractObserver | None = None,
        *,
        rngs: nnx.Rngs | None = None,
    ):
        """Initialize Cash contract.

        Args:
            attributes: Contract attributes
            risk_factor_observer: Observer for market data
            child_contract_observer: Observer for child contracts (optional)
            rngs: Random number generators (optional)

        Raises:
            ValueError: If validation fails

        Validations:
            - contract_type must be ContractType.CSH
            - notional_principal (NT) must be defined
            - contract_role (CNTRL) must be defined
            - currency (CUR) must be defined
        """
        super().__init__(
            attributes=attributes,
            risk_factor_observer=risk_factor_observer,
            child_contract_observer=child_contract_observer,
            rngs=rngs,
        )

        # Validate contract type
        if attributes.contract_type != ContractType.CSH:
            raise ValueError(f"Contract type must be CSH, got {attributes.contract_type}")

        # Validate required attributes for CSH
        if attributes.notional_principal is None:
            raise ValueError("CSH contract requires notional_principal (NT)")

        if attributes.contract_role is None:
            raise ValueError("CSH contract requires contract_role (CNTRL)")

        if attributes.currency is None:
            raise ValueError("CSH contract requires currency (CUR)")

    def generate_event_schedule(self) -> EventSchedule:
        """Generate CSH event schedule.

        CSH contracts only have Analysis Date (AD) events. A single AD event
        is created at the status date.

        Returns:
            EventSchedule with one AD event

        ACTUS Reference:
            CSH Contract Schedule:
                - AD event at status_date
                - No other events
        """
        # CSH only has a single AD event at status date
        events = [
            ContractEvent(
                event_type=EventType.AD,
                event_time=self.attributes.status_date,
                payoff=jnp.array(0.0, dtype=jnp.float32),
                currency=self.attributes.currency or "XXX",
                state_pre=None,
                state_post=None,
                sequence=0,
            )
        ]

        return EventSchedule(
            events=tuple(events),
            contract_id=self.attributes.contract_id,
        )

    def initialize_state(self) -> ContractState:
        """Initialize CSH contract state.

        CSH state is minimal:
        - nt: Notional with role sign applied: R(CNTRL) × NT
        - sd: Status date
        - All other states are zero or null

        Returns:
            Initial contract state

        ACTUS Reference:
            CSH State Initialization:
                nt = R(CNTRL) × NT
                sd = status_date
                All other states = 0.0 or null
        """
        # Get role sign: RPA/RFL = +1, RPL/RFL = -1, etc.
        role_sign = self._get_role_sign()

        # Notional with role sign
        notional = self.attributes.notional_principal or 0.0
        nt = role_sign * notional

        return ContractState(
            sd=self.attributes.status_date,
            tmd=self.attributes.status_date,  # No maturity for CSH
            nt=jnp.array(nt, dtype=jnp.float32),
            ipnr=jnp.array(0.0, dtype=jnp.float32),
            ipac=jnp.array(0.0, dtype=jnp.float32),
            feac=jnp.array(0.0, dtype=jnp.float32),
            nsc=jnp.array(1.0, dtype=jnp.float32),
            isc=jnp.array(1.0, dtype=jnp.float32),
        )

    def get_payoff_function(self, event_type: Any) -> CashPayoffFunction:
        """Get payoff function for CSH events.

        Args:
            event_type: Type of event (only AD for CSH)

        Returns:
            CashPayoffFunction instance
        """
        return CashPayoffFunction(
            contract_role=self.attributes.contract_role,
            currency=self.attributes.currency,
            settlement_currency=None,  # CSH doesn't use settlement currency
        )

    def get_state_transition_function(self, event_type: Any) -> CashStateTransitionFunction:
        """Get state transition function for CSH events.

        Args:
            event_type: Type of event (only AD for CSH)

        Returns:
            CashStateTransitionFunction instance
        """
        return CashStateTransitionFunction()

    def _get_role_sign(self) -> float:
        """Get the sign for contract role.

        Returns:
            +1.0 for RPA/RFL, -1.0 for RPL/RFL, etc.
        """
        from jactus.core import ContractRole

        if self.attributes.contract_role in (ContractRole.RPA, ContractRole.RFL):
            return 1.0
        if self.attributes.contract_role in (ContractRole.RPL, ContractRole.RFL):
            return -1.0
        # Default to 1.0 for other roles
        return 1.0
