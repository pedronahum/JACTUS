"""Future Contract (FUTUR) implementation.

This module implements the FUTUR contract type - a futures contract with
linear payoff based on the difference between spot and futures price.

ACTUS Reference:
    ACTUS v1.1 Section 7.16 - FUTUR: Future

Key Features:
    - Linear payoff: Xa = S_t - PFUT (can be positive or negative)
    - No premium payment (unlike options)
    - Always settles at maturity
    - Mark-to-market settlement
    - Underlier reference via contract_structure (CTST)

Differences from Options:
    - No exercise decision logic (always settles)
    - Linear payoff (not max-based like options)
    - No option premium
    - Settlement amount can be negative

Example:
    >>> from jactus.contracts.futur import FutureContract
    >>> from jactus.core import ContractAttributes, ContractType, ContractRole
    >>> from jactus.observers import ConstantRiskFactorObserver
    >>>
    >>> # Gold futures contract at $1800/oz
    >>> attrs = ContractAttributes(
    ...     contract_id="FUT-GOLD-001",
    ...     contract_type=ContractType.FUTUR,
    ...     contract_role=ContractRole.RPA,
    ...     status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
    ...     maturity_date=ActusDateTime(2024, 12, 31, 0, 0, 0),
    ...     currency="USD",
    ...     notional_principal=100.0,  # 100 ounces
    ...     future_price=1800.0,  # Agreed futures price
    ...     contract_structure="GC",  # Underlier (gold)
    ... )
    >>>
    >>> # Risk factor observer for gold price
    >>> rf_obs = ConstantRiskFactorObserver(constant_value=1850.0)  # Spot at $1850
    >>> contract = FutureContract(
    ...     attributes=attrs,
    ...     risk_factor_observer=rf_obs
    ... )
    >>> result = contract.simulate()
"""

from typing import Any

import jax.numpy as jnp

from jactus.contracts.base import BaseContract
from jactus.contracts.utils.underlier_valuation import get_underlier_market_value
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


class FuturePayoffFunction(BasePayoffFunction):
    """Payoff function for FUTUR contracts.

    Implements all FUTUR payoff functions according to ACTUS specification.

    ACTUS Reference:
        ACTUS v1.1 Section 7.16 - FUTUR Payoff Functions

    Events:
        AD: Analysis Date (0.0)
        PRD: Purchase Date (zero - no premium)
        TD: Termination Date (receive termination price)
        MD: Maturity Date (settlement amount calculated)
        STD: Settlement Date (receive settlement amount Xa)
        CE: Credit Event (contract default)

    State Variables Used:
        xa: Settlement amount (calculated at MD) = S_t - PFUT
        prf: Contract performance (default status)
    """

    def calculate_payoff(
        self,
        event_type: Any,
        state: ContractState,
        attributes: ContractAttributes,
        time: ActusDateTime,
        risk_factor_observer: RiskFactorObserver,
    ) -> jnp.ndarray:
        """Calculate payoff for FUTUR events.

        Dispatches to specific payoff function based on event type.

        Args:
            event_type: Type of event
            state: Current contract state
            attributes: Contract attributes
            time: Event time
            risk_factor_observer: Risk factor observer

        Returns:
            Payoff amount as JAX array
        """
        # Map event types to payoff functions
        if event_type == EventType.AD:
            return self._pof_ad(state, attributes, time, risk_factor_observer)
        if event_type == EventType.IED:
            return self._pof_ied(state, attributes, time, risk_factor_observer)
        if event_type == EventType.PRD:
            return self._pof_prd(state, attributes, time, risk_factor_observer)
        if event_type == EventType.TD:
            return self._pof_td(state, attributes, time, risk_factor_observer)
        if event_type == EventType.MD:
            return self._pof_md(state, attributes, time, risk_factor_observer)
        if event_type == EventType.STD:
            return self._pof_std(state, attributes, time, risk_factor_observer)
        if event_type == EventType.CE:
            return self._pof_ce(state, attributes, time, risk_factor_observer)
        # Unknown event type, return zero
        return jnp.array(0.0, dtype=jnp.float32)

    def _pof_ad(
        self,
        state: ContractState,
        attributes: ContractAttributes,
        time: ActusDateTime,
        risk_factor_observer: RiskFactorObserver,
    ) -> jnp.ndarray:
        """POF_AD_FUTUR: Analysis Date payoff.

        Analysis dates have zero payoff.

        Returns:
            0.0
        """
        return jnp.array(0.0, dtype=jnp.float32)

    def _pof_ied(
        self,
        state: ContractState,
        attributes: ContractAttributes,
        time: ActusDateTime,
        risk_factor_observer: RiskFactorObserver,
    ) -> jnp.ndarray:
        """POF_IED_FUTUR: Initial Exchange Date payoff.

        Not used for FUTUR (futures start at PRD or status date).

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
        """POF_PRD_FUTUR: Purchase Date payoff.

        No premium for futures (zero cashflow).

        Returns:
            0.0
        """
        return jnp.array(0.0, dtype=jnp.float32)

    def _pof_td(
        self,
        state: ContractState,
        attributes: ContractAttributes,
        time: ActusDateTime,
        risk_factor_observer: RiskFactorObserver,
    ) -> jnp.ndarray:
        """POF_TD_FUTUR: Termination Date payoff.

        Receive termination price (positive cashflow for seller).

        Formula:
            POF_TD = PTD × NT

        Returns:
            Positive termination payment
        """
        ptd = attributes.price_at_termination_date or 0.0
        nt = attributes.notional_principal or 1.0

        return jnp.array(ptd * nt, dtype=jnp.float32)

    def _pof_md(
        self,
        state: ContractState,
        attributes: ContractAttributes,
        time: ActusDateTime,
        risk_factor_observer: RiskFactorObserver,
    ) -> jnp.ndarray:
        """POF_MD_FUTUR: Maturity Date payoff.

        Zero payoff at maturity (settlement at STD).

        Returns:
            0.0
        """
        return jnp.array(0.0, dtype=jnp.float32)

    def _pof_std(
        self,
        state: ContractState,
        attributes: ContractAttributes,
        time: ActusDateTime,
        risk_factor_observer: RiskFactorObserver,
    ) -> jnp.ndarray:
        """POF_STD_FUTUR: Settlement Date payoff.

        Receive settlement amount (can be positive or negative).

        Formula:
            POF_STD = Xa × NT

        where Xa = S_t - PFUT (linear payoff, not max-based).

        Returns:
            Settlement amount (positive or negative)
        """
        xa = float(state.xa) if hasattr(state, "xa") else 0.0
        nt = attributes.notional_principal or 1.0

        return jnp.array(xa * nt, dtype=jnp.float32)

    def _pof_ce(
        self,
        state: ContractState,
        attributes: ContractAttributes,
        time: ActusDateTime,
        risk_factor_observer: RiskFactorObserver,
    ) -> jnp.ndarray:
        """POF_CE_FUTUR: Credit Event payoff.

        Zero payoff on credit event (contract worthless if counterparty defaults).

        Returns:
            0.0
        """
        return jnp.array(0.0, dtype=jnp.float32)


class FutureStateTransitionFunction(BaseStateTransitionFunction):
    """State transition function for FUTUR contracts.

    Handles state transitions for futures contracts, including:
    - Settlement amount calculation (linear payoff)
    - State updates after settlement
    """

    def transition_state(
        self,
        event_type: EventType,
        state_pre: ContractState,
        attributes: ContractAttributes,
        time: ActusDateTime,
        risk_factor_observer: RiskFactorObserver,
    ) -> ContractState:
        """Calculate state transition for a given event.

        Args:
            event_type: Type of event triggering the transition
            state_pre: Current contract state (before event)
            attributes: Contract attributes
            time: Event time
            risk_factor_observer: Observer for market data

        Returns:
            Updated contract state (after event)
        """
        # Create a dummy event for compatibility with helper methods
        event = ContractEvent(
            event_type=event_type,
            event_time=time,
            payoff=0.0,
            currency=attributes.currency,
            sequence=0,
        )

        if event_type == EventType.MD:
            return self._stf_md(state_pre, event, attributes, risk_factor_observer)
        if event_type == EventType.STD:
            return self._stf_std(state_pre, event, attributes)
        # No state change for other events
        return state_pre

    def _stf_md(
        self,
        state: ContractState,
        event: ContractEvent,
        attributes: ContractAttributes,
        risk_factor_observer: RiskFactorObserver,
    ) -> ContractState:
        """STF_MD_FUTUR: Maturity Date state transition.

        Calculate settlement amount based on spot price vs futures price.

        Formula:
            Xa = S_t - PFUT (linear payoff, can be positive or negative)

        Returns:
            State with updated xa (settlement amount)
        """
        # Get underlier price
        underlier_ref = attributes.contract_structure
        if underlier_ref is None:
            raise ValueError("contract_structure (underlier) required for FUTUR")

        spot_price = get_underlier_market_value(
            underlier_ref, event.event_time, risk_factor_observer
        )

        # Get futures price
        futures_price = attributes.future_price
        if futures_price is None:
            raise ValueError("future_price (PFUT) required for FUTUR")

        # Calculate settlement amount: Xa = S_t - PFUT (linear, can be negative)
        settlement_amount = float(spot_price) - futures_price

        # Update state with settlement amount
        return ContractState(
            sd=state.sd,
            tmd=state.tmd,
            nt=state.nt,
            ipnr=state.ipnr,
            ipac=state.ipac,
            feac=state.feac,
            nsc=state.nsc,
            isc=state.isc,
            prf=state.prf,
            xa=jnp.array(settlement_amount, dtype=jnp.float32),
        )

    def _stf_std(
        self,
        state: ContractState,
        event: ContractEvent,
        attributes: ContractAttributes,
    ) -> ContractState:
        """STF_STD_FUTUR: Settlement Date state transition.

        Reset settlement amount after settlement.

        Returns:
            State with xa reset to 0
        """
        return ContractState(
            sd=state.sd,
            tmd=state.tmd,
            nt=state.nt,
            ipnr=state.ipnr,
            ipac=state.ipac,
            feac=state.feac,
            nsc=state.nsc,
            isc=state.isc,
            prf=state.prf,
            xa=jnp.array(0.0, dtype=jnp.float32),
        )


class FutureContract(BaseContract):
    """Future Contract (FUTUR) implementation.

    Represents a futures contract with linear payoff based on spot vs futures price.

    Attributes:
        future_price (PFUT): Agreed futures price
        contract_structure (CTST): Underlier reference
        notional_principal (NT): Number of units (e.g., contracts)
        maturity_date (MD): Futures expiration date

    Key Differences from Options:
        - Linear payoff (not max-based)
        - No premium payment
        - Settlement amount can be negative
        - Always settles at maturity (no exercise decision)

    Example:
        >>> # S&P 500 futures contract
        >>> attrs = ContractAttributes(
        ...     contract_type=ContractType.FUTUR,
        ...     future_price=4500.0,
        ...     contract_structure="SPX",
        ...     notional_principal=1.0,
        ...     maturity_date=ActusDateTime(2024, 12, 31, 0, 0, 0),
        ...     ...
        ... )
        >>> contract = FutureContract(attrs, rf_obs)
        >>> events = contract.generate_event_schedule()
    """

    def __init__(
        self,
        attributes: ContractAttributes,
        risk_factor_observer: RiskFactorObserver,
        child_contract_observer: ChildContractObserver | None = None,
    ):
        """Initialize FUTUR contract.

        Args:
            attributes: Contract attributes
            risk_factor_observer: Risk factor observer for market data
            child_contract_observer: Observer for underlier contracts (optional)

        Raises:
            ValueError: If validation fails
        """
        # Validate contract type
        if attributes.contract_type != ContractType.FUTUR:
            raise ValueError(f"Expected contract_type=FUTUR, got {attributes.contract_type}")

        # Validate futures price
        if attributes.future_price is None:
            raise ValueError("future_price (PFUT) is required for FUTUR")

        # Validate underlier reference
        if attributes.contract_structure is None:
            raise ValueError("contract_structure (CTST) required for FUTUR (underlier reference)")

        # Validate maturity date
        if attributes.maturity_date is None:
            raise ValueError("maturity_date is required for FUTUR")

        super().__init__(attributes, risk_factor_observer, child_contract_observer)

    def generate_event_schedule(self) -> EventSchedule:
        """Generate event schedule for FUTUR contract.

        Events:
        - AD (optional): Analysis dates
        - PRD (optional): Purchase date (zero payment)
        - TD (optional): Termination date
        - MD: Maturity date (settlement amount calculated)
        - STD: Settlement date (receive settlement amount)

        Returns:
            Event schedule with all contract events
        """
        events = []

        # Analysis dates (if specified)
        if self.attributes.analysis_dates:
            for ad_time in self.attributes.analysis_dates:
                events.append(
                    ContractEvent(
                        event_type=EventType.AD,
                        event_time=ad_time,
                        payoff=0.0,
                        currency=self.attributes.currency,
                        sequence=0,
                    )
                )

        # Purchase date (if specified)
        if self.attributes.purchase_date:
            events.append(
                ContractEvent(
                    event_type=EventType.PRD,
                    event_time=self.attributes.purchase_date,
                    payoff=0.0,  # No premium for futures
                    currency=self.attributes.currency,
                    sequence=1,
                )
            )

        # Termination date (if specified)
        if self.attributes.termination_date:
            events.append(
                ContractEvent(
                    event_type=EventType.TD,
                    event_time=self.attributes.termination_date,
                    payoff=0.0,
                    currency=self.attributes.currency,
                    sequence=2,
                )
            )

        # Maturity date
        events.append(
            ContractEvent(
                event_type=EventType.MD,
                event_time=self.attributes.maturity_date,
                payoff=0.0,
                currency=self.attributes.currency,
                sequence=3,
            )
        )

        # Settlement date (same as maturity for futures)
        settlement_date = self.attributes.maturity_date
        events.append(
            ContractEvent(
                event_type=EventType.STD,
                event_time=settlement_date,
                payoff=0.0,
                currency=self.attributes.currency,
                sequence=4,
            )
        )

        # Sort events by time and sequence
        events.sort(key=lambda e: (e.event_time.to_iso(), e.sequence))

        return EventSchedule(
            contract_id=self.attributes.contract_id,
            events=events,
        )

    def initialize_state(self) -> ContractState:
        """Initialize contract state at status date.

        Returns:
            Initial contract state with xa=0
        """
        prf = self.attributes.contract_performance
        if prf is None:
            prf = "PF"  # Default: performing

        return ContractState(
            sd=self.attributes.status_date,
            tmd=self.attributes.maturity_date,
            nt=jnp.array(0.0, dtype=jnp.float32),
            ipnr=jnp.array(0.0, dtype=jnp.float32),
            ipac=jnp.array(0.0, dtype=jnp.float32),
            feac=jnp.array(0.0, dtype=jnp.float32),
            nsc=jnp.array(1.0, dtype=jnp.float32),
            isc=jnp.array(1.0, dtype=jnp.float32),
            prf=prf,
            xa=jnp.array(0.0, dtype=jnp.float32),  # Settlement amount
        )

    def get_payoff_function(self, event_type: Any) -> FuturePayoffFunction:
        """Get payoff function for FUTUR contract.

        Args:
            event_type: The event type (for compatibility with BaseContract)

        Returns:
            FuturePayoffFunction instance
        """
        return FuturePayoffFunction(
            contract_role=self.attributes.contract_role,
            currency=self.attributes.currency,
        )

    def get_state_transition_function(self, event_type: Any) -> FutureStateTransitionFunction:
        """Get state transition function for FUTUR contract.

        Args:
            event_type: The event type (for compatibility with BaseContract)

        Returns:
            FutureStateTransitionFunction instance
        """
        return FutureStateTransitionFunction()
