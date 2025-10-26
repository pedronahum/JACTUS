"""Commodity (COM) contract implementation.

This module implements the COM contract type - a commodity position with price exposure.
COM is a simple contract representing commodity ownership (e.g., gold, oil, wheat).

ACTUS Reference:
    ACTUS v1.1 Section 7.10 - COM: Commodity

Key Features:
    - Commodity position value from market observation
    - Purchase and termination events
    - Minimal state (only performance and status date)
    - 4 event types: AD, PRD, TD, CE

Example:
    >>> from jactus.contracts.com import CommodityContract
    >>> from jactus.core import ContractAttributes, ContractType, ContractRole
    >>> from jactus.observers import ConstantRiskFactorObserver
    >>>
    >>> attrs = ContractAttributes(
    ...     contract_id="COM-001",
    ...     contract_type=ContractType.COM,
    ...     contract_role=ContractRole.RPA,
    ...     status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
    ...     currency="USD",
    ...     price_at_purchase_date=7500.0,  # Total purchase price
    ...     price_at_termination_date=8200.0,  # Total sale price
    ... )
    >>>
    >>> rf_obs = ConstantRiskFactorObserver(constant_value=80.0)
    >>> contract = CommodityContract(
    ...     attributes=attrs,
    ...     risk_factor_observer=rf_obs
    ... )
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


class CommodityPayoffFunction(BasePayoffFunction):
    """Payoff function for COM contracts.

    Implements all COM payoff functions according to ACTUS specification.

    ACTUS Reference:
        ACTUS v1.1 Section 7.10 - COM Payoff Functions

    Events:
        AD: Analysis Date (0.0)
        PRD: Purchase Date (pay purchase price)
        TD: Termination Date (receive termination price)
        CE: Credit Event (0.0)
    """

    def calculate_payoff(
        self,
        event_type: Any,
        state: ContractState,
        attributes: ContractAttributes,
        time: ActusDateTime,
        risk_factor_observer: RiskFactorObserver,
    ) -> jnp.ndarray:
        """Calculate payoff for COM events.

        Dispatches to specific payoff function based on event type.

        Args:
            event_type: Type of event
            state: Current contract state
            attributes: Contract attributes
            time: Event time
            risk_factor_observer: Observer for market data

        Returns:
            Payoff amount as JAX array

        ACTUS Reference:
            POF_[event]_COM functions from Section 7.10
        """
        if event_type == EventType.AD:
            return self._pof_ad(state, attributes, time, risk_factor_observer)
        if event_type == EventType.PRD:
            return self._pof_prd(state, attributes, time, risk_factor_observer)
        if event_type == EventType.TD:
            return self._pof_td(state, attributes, time, risk_factor_observer)
        if event_type == EventType.CE:
            return self._pof_ce(state, attributes, time, risk_factor_observer)
        # Unknown event type - return 0
        return jnp.array(0.0, dtype=jnp.float32)

    def _pof_ad(
        self,
        state: ContractState,
        attributes: ContractAttributes,
        time: ActusDateTime,
        risk_factor_observer: RiskFactorObserver,
    ) -> jnp.ndarray:
        """POF_AD_COM: Analysis Date has no cashflow.

        Returns:
            0.0
        """
        return jnp.array(0.0, dtype=jnp.float32)

    def _pof_prd(
        self,
        state: ContractState,
        attributes: ContractAttributes,
        time: ActusDateTime,
        risk_factor_observer: RiskFactorObserver,
    ) -> jnp.ndarray:
        """POF_PRD_COM: Purchase Date - pay purchase price.

        Formula:
            POF_PRD_COM = X^CURS_CUR(t) × R(CNTRL) × (-PPRD)

        Where:
            PPRD: Price at purchase date (total price, not per unit)
            R(CNTRL): Role sign
            X^CURS_CUR(t): FX rate

        Note:
            For COM, price_at_purchase_date represents the total purchase price.
            If you need per-unit pricing, multiply unit price by quantity before
            setting this attribute.

        Returns:
            Negative of purchase price (outflow for buyer)
        """
        # Get purchase price (total amount)
        pprd = attributes.price_at_purchase_date or 0.0

        # Purchase is negative cashflow (paying for commodity)
        payoff = -pprd

        return jnp.array(payoff, dtype=jnp.float32)

    def _pof_td(
        self,
        state: ContractState,
        attributes: ContractAttributes,
        time: ActusDateTime,
        risk_factor_observer: RiskFactorObserver,
    ) -> jnp.ndarray:
        """POF_TD_COM: Termination Date - receive termination price.

        Formula:
            POF_TD_COM = X^CURS_CUR(t) × R(CNTRL) × PTD

        Where:
            PTD: Price at termination date (total price, not per unit)
            R(CNTRL): Role sign
            X^CURS_CUR(t): FX rate

        Note:
            For COM, price_at_termination_date represents the total sale price.
            If you need per-unit pricing, multiply unit price by quantity before
            setting this attribute.

        Returns:
            Termination price (inflow for seller)
        """
        # Get termination price (total amount)
        ptd = attributes.price_at_termination_date or 0.0

        # Termination is positive cashflow (receiving sale proceeds)
        payoff = ptd

        return jnp.array(payoff, dtype=jnp.float32)

    def _pof_ce(
        self,
        state: ContractState,
        attributes: ContractAttributes,
        time: ActusDateTime,
        risk_factor_observer: RiskFactorObserver,
    ) -> jnp.ndarray:
        """POF_CE_COM: Credit Event - no cashflow.

        Returns:
            0.0 (credit events handled in state transition)
        """
        return jnp.array(0.0, dtype=jnp.float32)


class CommodityStateTransitionFunction(BaseStateTransitionFunction):
    """State transition function for COM contracts.

    Implements all COM state transition functions according to ACTUS specification.

    ACTUS Reference:
        ACTUS v1.1 Section 7.10 - COM State Transition Functions

    Note:
        COM has minimal state - only status date (sd) and performance (prf).
        All events simply update the status date.
    """

    def transition_state(
        self,
        event_type: Any,
        state_pre: ContractState,
        attributes: ContractAttributes,
        time: ActusDateTime,
        risk_factor_observer: RiskFactorObserver,
    ) -> ContractState:
        """Transition COM contract state.

        All COM events have the same state transition: update status date only.

        Args:
            event_type: Type of event
            state_pre: State before event
            attributes: Contract attributes
            time: Event time
            risk_factor_observer: Observer for market data

        Returns:
            Updated contract state

        ACTUS Reference:
            STF_[event]_COM functions from Section 7.10
        """
        # All COM events just update status date
        # Performance tracking could be added here if needed
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


class CommodityContract(BaseContract):
    """Commodity (COM) contract implementation.

    Represents a commodity position with price exposure.
    COM is one of the simplest ACTUS contracts, similar to STK but for
    physical or financial commodities (gold, oil, wheat, etc.).

    ACTUS Reference:
        ACTUS v1.1 Section 7.10

    Attributes:
        attributes: Contract terms and parameters
        risk_factor_observer: Observer for market prices
        child_contract_observer: Observer for child contracts (optional)

    Example:
        >>> attrs = ContractAttributes(
        ...     contract_id="COM-001",
        ...     contract_type=ContractType.COM,
        ...     contract_role=ContractRole.RPA,
        ...     status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
        ...     currency="USD",
        ...     price_at_purchase_date=7500.0,
        ...     price_at_termination_date=8200.0,
        ... )
        >>> contract = CommodityContract(attrs, risk_obs)
        >>> result = contract.simulate()
    """

    def __init__(
        self,
        attributes: ContractAttributes,
        risk_factor_observer: RiskFactorObserver,
        child_contract_observer: ChildContractObserver | None = None,
        rngs: nnx.Rngs | None = None,
    ):
        """Initialize COM contract.

        Args:
            attributes: Contract attributes
            risk_factor_observer: Observer for market data
            child_contract_observer: Optional observer for child contracts
            rngs: Optional Flax NNX random number generators

        Raises:
            ValueError: If contract_type is not COM or required attributes missing
        """
        super().__init__(
            attributes=attributes,
            risk_factor_observer=risk_factor_observer,
            child_contract_observer=child_contract_observer,
            rngs=rngs,
        )

        # Validate contract type
        if attributes.contract_type != ContractType.COM:
            raise ValueError(f"Contract type must be COM, got {attributes.contract_type}")

        # COM doesn't have strict requirements beyond contract_type
        # price attributes are recommended but not required

    def generate_event_schedule(self) -> EventSchedule:
        """Generate COM event schedule.

        Generates events for commodity contract:
        - AD: Analysis dates (if specified)
        - PRD: Purchase date (if specified)
        - TD: Termination date (if specified)

        Returns:
            EventSchedule with all contract events

        ACTUS Reference:
            COM Contract Schedule from Section 7.10
        """
        events: list[ContractEvent] = []

        # PRD: Purchase Date (if defined)
        if self.attributes.purchase_date:
            events.append(
                ContractEvent(
                    event_type=EventType.PRD,
                    event_time=self.attributes.purchase_date,
                    payoff=jnp.array(0.0, dtype=jnp.float32),
                    currency=self.attributes.currency or "XXX",
                    state_pre=None,
                    state_post=None,
                    sequence=len(events),
                )
            )

        # TD: Termination Date (if defined)
        if self.attributes.termination_date:
            events.append(
                ContractEvent(
                    event_type=EventType.TD,
                    event_time=self.attributes.termination_date,
                    payoff=jnp.array(0.0, dtype=jnp.float32),
                    currency=self.attributes.currency or "XXX",
                    state_pre=None,
                    state_post=None,
                    sequence=len(events),
                )
            )

        # Sort events by time
        events.sort(key=lambda e: (e.event_time.to_iso(), e.sequence))

        # Reassign sequence numbers
        for i, event in enumerate(events):
            events[i] = ContractEvent(
                event_type=event.event_type,
                event_time=event.event_time,
                payoff=event.payoff,
                currency=event.currency,
                state_pre=event.state_pre,
                state_post=event.state_post,
                sequence=i,
            )

        return EventSchedule(
            events=tuple(events),
            contract_id=self.attributes.contract_id,
        )

    def initialize_state(self) -> ContractState:
        """Initialize COM contract state.

        COM has minimal state - only status date and performance.

        ACTUS Reference:
            COM State Initialization from Section 7.10

        Returns:
            Initial contract state
        """
        # COM has minimal state - just status date
        return ContractState(
            sd=self.attributes.status_date,
            tmd=self.attributes.termination_date or self.attributes.status_date,
            nt=jnp.array(0.0, dtype=jnp.float32),
            ipnr=jnp.array(0.0, dtype=jnp.float32),
            ipac=jnp.array(0.0, dtype=jnp.float32),
            feac=jnp.array(0.0, dtype=jnp.float32),
            nsc=jnp.array(1.0, dtype=jnp.float32),
            isc=jnp.array(1.0, dtype=jnp.float32),
        )

    def get_payoff_function(self, event_type: Any) -> CommodityPayoffFunction:
        """Get payoff function for COM events."""
        return CommodityPayoffFunction(
            contract_role=self.attributes.contract_role,
            currency=self.attributes.currency,
            settlement_currency=None,
        )

    def get_state_transition_function(self, event_type: Any) -> CommodityStateTransitionFunction:
        """Get state transition function for COM events."""
        return CommodityStateTransitionFunction()
