"""Stock (STK) contract implementation.

This module implements the STK contract type - an equity position with dividend payments.
STK is a simple contract representing stock ownership with potential dividend income.

ACTUS Reference:
    ACTUS v1.1 Section 7.9 - STK: Stock

Key Features:
    - Equity position value from market observation
    - Fixed or market-observed dividend payments
    - Purchase and termination events
    - Minimal state (only performance and status date)
    - 6 event types total

Example:
    >>> from jactus.contracts.stk import StockContract
    >>> from jactus.core import ContractAttributes, ContractType, ContractRole
    >>> from jactus.observers import ConstantRiskFactorObserver
    >>>
    >>> attrs = ContractAttributes(
    ...     contract_id="STK-001",
    ...     contract_type=ContractType.STK,
    ...     contract_role=ContractRole.RPA,
    ...     status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
    ...     currency="USD",
    ...     market_object_code="AAPL",  # Stock ticker
    ... )
    >>>
    >>> rf_obs = ConstantRiskFactorObserver(constant_value=150.0)  # Stock price
    >>> contract = StockContract(
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
from jactus.utilities import contract_role_sign


class StockPayoffFunction(BasePayoffFunction):
    """Payoff function for STK contracts.

    Implements all 6 STK payoff functions according to ACTUS specification.

    ACTUS Reference:
        ACTUS v1.1 Section 7.9 - STK Payoff Functions

    Events:
        AD: Analysis Date (0.0)
        PRD: Purchase Date (pay purchase price)
        TD: Termination Date (receive termination price)
        DV(fix): Fixed Dividend Payment
        DV: Market-Observed Dividend Payment
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
        """Calculate payoff for STK events.

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
            POF_[event]_STK functions from Section 7.9
        """
        if event_type == EventType.AD:
            return self._pof_ad(state, attributes, time, risk_factor_observer)
        if event_type == EventType.PRD:
            return self._pof_prd(state, attributes, time, risk_factor_observer)
        if event_type == EventType.TD:
            return self._pof_td(state, attributes, time, risk_factor_observer)
        if event_type == EventType.DV:
            return self._pof_dv(state, attributes, time, risk_factor_observer)
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
        """POF_AD_STK: Analysis Date has no cashflow.

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
        """POF_PRD_STK: Purchase Date - pay purchase price.

        Formula:
            POF_PRD_STK = X^CURS_CUR(t) × R(CNTRL) × (-PPRD)

        Where:
            PPRD: Price at purchase date
            R(CNTRL): Role sign
            X^CURS_CUR(t): FX rate

        Returns:
            Negative of purchase price (outflow for buyer)
        """
        pprd = attributes.price_at_purchase_date or 0.0
        role_sign = contract_role_sign(attributes.contract_role)

        payoff = role_sign * (-pprd)

        return jnp.array(payoff, dtype=jnp.float32)

    def _pof_td(
        self,
        state: ContractState,
        attributes: ContractAttributes,
        time: ActusDateTime,
        risk_factor_observer: RiskFactorObserver,
    ) -> jnp.ndarray:
        """POF_TD_STK: Termination Date - receive termination price.

        Formula:
            POF_TD_STK = X^CURS_CUR(t) × R(CNTRL) × PTD

        Where:
            PTD: Price at termination date
            R(CNTRL): Role sign
            X^CURS_CUR(t): FX rate

        Returns:
            Termination price (inflow for seller)
        """
        ptd = attributes.price_at_termination_date or 0.0
        role_sign = contract_role_sign(attributes.contract_role)

        payoff = role_sign * ptd

        return jnp.array(payoff, dtype=jnp.float32)

    def _pof_dv(
        self,
        state: ContractState,
        attributes: ContractAttributes,
        time: ActusDateTime,
        risk_factor_observer: RiskFactorObserver,
    ) -> jnp.ndarray:
        """POF_DV_STK: Dividend Payment.

        Formula (observed):
            POF_DV_STK = R(CNTRL) × O_dv(DVMO, t)

        Where:
            O_dv: Observed dividend amount from risk factor
            R(CNTRL): Role sign
        """
        role_sign = contract_role_sign(attributes.contract_role)

        # Observe dividend amount from risk factors
        dvmo = attributes.market_object_code_of_dividends or ""
        if dvmo:
            dv_amount = float(risk_factor_observer.observe_risk_factor(dvmo, time))
        else:
            dv_amount = 0.0

        return jnp.array(role_sign * dv_amount, dtype=jnp.float32)

    def _pof_ce(
        self,
        state: ContractState,
        attributes: ContractAttributes,
        time: ActusDateTime,
        risk_factor_observer: RiskFactorObserver,
    ) -> jnp.ndarray:
        """POF_CE_STK: Credit Event - no cashflow.

        Returns:
            0.0 (credit events handled in state transition)
        """
        return jnp.array(0.0, dtype=jnp.float32)


class StockStateTransitionFunction(BaseStateTransitionFunction):
    """State transition function for STK contracts.

    Implements all 6 STK state transition functions according to ACTUS specification.

    ACTUS Reference:
        ACTUS v1.1 Section 7.9 - STK State Transition Functions

    Note:
        STK has minimal state - only status date (sd) and performance (prf).
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
        """Transition STK contract state.

        All STK events have the same state transition: update status date only.

        Args:
            event_type: Type of event
            state_pre: State before event
            attributes: Contract attributes
            time: Event time
            risk_factor_observer: Observer for market data

        Returns:
            Updated contract state

        ACTUS Reference:
            STF_[event]_STK functions from Section 7.9
        """
        # All STK events just update status date
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


class StockContract(BaseContract):
    """Stock (STK) contract implementation.

    Represents an equity position with potential dividend payments.
    STK is one of the simplest ACTUS contracts, with minimal state
    and straightforward cashflow logic.

    ACTUS Reference:
        ACTUS v1.1 Section 7.9

    Attributes:
        attributes: Contract terms and parameters
        risk_factor_observer: Observer for market prices and dividends
        child_contract_observer: Observer for child contracts (optional)

    Example:
        >>> attrs = ContractAttributes(
        ...     contract_id="STK-001",
        ...     contract_type=ContractType.STK,
        ...     contract_role=ContractRole.RPA,
        ...     status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
        ...     currency="USD",
        ...     market_object_code="AAPL",
        ... )
        >>> contract = StockContract(attrs, risk_obs)
        >>> result = contract.simulate()
    """

    def __init__(
        self,
        attributes: ContractAttributes,
        risk_factor_observer: RiskFactorObserver,
        child_contract_observer: ChildContractObserver | None = None,
        rngs: nnx.Rngs | None = None,
    ):
        """Initialize STK contract.

        Args:
            attributes: Contract attributes
            risk_factor_observer: Observer for market data
            child_contract_observer: Optional observer for child contracts
            rngs: Optional Flax NNX random number generators

        Raises:
            ValueError: If contract_type is not STK or required attributes missing
        """
        super().__init__(
            attributes=attributes,
            risk_factor_observer=risk_factor_observer,
            child_contract_observer=child_contract_observer,
            rngs=rngs,
        )

        # Validate contract type
        if attributes.contract_type != ContractType.STK:
            raise ValueError(f"Contract type must be STK, got {attributes.contract_type}")

        # STK doesn't have strict requirements beyond contract_type
        # market_object_code is recommended for price observation but not required

    def _apply_bdc(self, date: ActusDateTime) -> ActusDateTime:
        """Apply business day convention adjustment to a date."""
        from jactus.utilities.calendars import MondayToFridayCalendar

        bdc = self.attributes.business_day_convention
        cal = self.attributes.calendar
        if not bdc or bdc == "NULL" or not cal or cal in ("NO_CALENDAR", "NC"):
            return date
        calendar = MondayToFridayCalendar()
        bdc_val = bdc.value if hasattr(bdc, "value") else str(bdc)
        if bdc_val in ("CSF", "SCF", "CSMF", "SCMF"):
            return calendar.next_business_day(date)
        if bdc_val in ("CSP", "SCP", "CSMP", "SCMP"):
            return calendar.previous_business_day(date)
        return date

    def generate_event_schedule(self) -> EventSchedule:
        """Generate STK event schedule.

        Generates events for stock contract:
        - AD: Analysis dates (if specified)
        - PRD: Purchase date (if specified)
        - TD: Termination date (if specified)
        - DV: Dividend events (if dividend schedule specified)

        Returns:
            EventSchedule with all contract events

        ACTUS Reference:
            STK Contract Schedule from Section 7.9
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

        # DV: Dividend events
        if self.attributes.dividend_cycle:
            from jactus.utilities.schedules import generate_schedule

            dv_start = (
                self.attributes.dividend_anchor
                or self.attributes.purchase_date
                or self.attributes.status_date
            )
            dv_end = self.attributes.termination_date or self.attributes.maturity_date
            if dv_end:
                dv_dates = generate_schedule(
                    start=dv_start,
                    cycle=self.attributes.dividend_cycle,
                    end=dv_end,
                )
                for dv_time in dv_dates:
                    if dv_time > self.attributes.status_date:
                        dv_time = self._apply_bdc(dv_time)
                        events.append(
                            ContractEvent(
                                event_type=EventType.DV,
                                event_time=dv_time,
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
        """Initialize STK contract state.

        STK has minimal state - only status date and performance.

        ACTUS Reference:
            STK State Initialization from Section 7.9

        Returns:
            Initial contract state
        """
        # STK has minimal state - just status date
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

    def get_payoff_function(self, event_type: Any) -> StockPayoffFunction:
        """Get payoff function for STK events."""
        return StockPayoffFunction(
            contract_role=self.attributes.contract_role,
            currency=self.attributes.currency,
            settlement_currency=None,
        )

    def get_state_transition_function(self, event_type: Any) -> StockStateTransitionFunction:
        """Get state transition function for STK events."""
        return StockStateTransitionFunction()
