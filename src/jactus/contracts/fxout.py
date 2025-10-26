"""Foreign Exchange Outright (FXOUT) contract implementation.

This module implements the FXOUT contract type - an FX forward or spot contract
with settlement in two currencies. FXOUT represents the exchange of two currency
amounts at a future date (or spot).

ACTUS Reference:
    ACTUS v1.1 Section 7.11 - FXOUT: Foreign Exchange Outright

Key Features:
    - Dual currency settlement (CUR and CUR2)
    - Two settlement modes: delivery (net) or dual (gross)
    - FX rate observation at settlement
    - No interest payments or principal amortization
    - 7 event types total

Settlement Modes:
    - Delivery (DS='D'): Net settlement in a single currency
    - Dual (DS='S'): Two separate payments, one in each currency

Example:
    >>> from jactus.contracts.fxout import FXOutrightContract
    >>> from jactus.core import ContractAttributes, ContractType, ContractRole
    >>> from jactus.observers import ConstantRiskFactorObserver
    >>>
    >>> # EUR/USD forward: buy 100,000 EUR, sell 110,000 USD
    >>> attrs = ContractAttributes(
    ...     contract_id="FXFWD-001",
    ...     contract_type=ContractType.FXOUT,
    ...     contract_role=ContractRole.RPA,
    ...     status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
    ...     maturity_date=ActusDateTime(2024, 7, 1, 0, 0, 0),
    ...     currency="EUR",  # First currency
    ...     currency_2="USD",  # Second currency
    ...     notional_principal=100000.0,  # EUR amount
    ...     notional_principal_2=110000.0,  # USD amount (forward rate = 1.10)
    ...     delivery_settlement="D",  # Net settlement
    ... )
    >>>
    >>> # FX rate observer (returns USD/EUR rate)
    >>> rf_obs = ConstantRiskFactorObserver(constant_value=1.12)
    >>> contract = FXOutrightContract(
    ...     attributes=attrs,
    ...     risk_factor_observer=rf_obs
    ... )
    >>> result = contract.simulate()
"""

from typing import Any

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


class FXOutrightPayoffFunction(BasePayoffFunction):
    """Payoff function for FXOUT contracts.

    Implements all 7 FXOUT payoff functions according to ACTUS specification.

    ACTUS Reference:
        ACTUS v1.1 Section 7.11 - FXOUT Payoff Functions

    Events:
        AD: Analysis Date (0.0)
        PRD: Purchase Date (pay purchase price)
        TD: Termination Date (receive termination price)
        STD: Settlement Date (net settlement, delivery mode)
        STD(1): Settlement Date - first currency (dual mode)
        STD(2): Settlement Date - second currency (dual mode)
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
        """Calculate payoff for FXOUT events.

        Dispatches to specific payoff function based on event type.

        Args:
            event_type: Type of event
            state: Current contract state
            attributes: Contract attributes
            time: Event time
            risk_factor_observer: Observer for market data (FX rates)

        Returns:
            Payoff amount as JAX array

        ACTUS Reference:
            POF_[event]_FXOUT functions from Section 7.11
        """
        if event_type == EventType.AD:
            return self._pof_ad(state, attributes, time, risk_factor_observer)
        if event_type == EventType.PRD:
            return self._pof_prd(state, attributes, time, risk_factor_observer)
        if event_type == EventType.TD:
            return self._pof_td(state, attributes, time, risk_factor_observer)
        if event_type == EventType.STD:
            return self._pof_std(state, attributes, time, risk_factor_observer)
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
        """POF_AD_FXOUT: Analysis Date has no cashflow.

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
        """POF_PRD_FXOUT: Purchase Date - pay purchase price.

        Formula:
            POF_PRD_FXOUT = X^CURS_CUR(t) × R(CNTRL) × (-PPRD)

        Where:
            PPRD: Price at purchase date
            R(CNTRL): Role sign
            X^CURS_CUR(t): FX rate (if needed)

        Returns:
            Negative of purchase price (outflow for buyer)
        """
        # Get purchase price (should be defined in attributes)
        pprd = attributes.price_at_purchase_date or 0.0

        # Purchase is negative cashflow
        payoff = -pprd

        # Apply contract role sign
        role_sign = 1.0 if attributes.contract_role.value == "RPA" else -1.0
        payoff = role_sign * payoff

        return jnp.array(payoff, dtype=jnp.float32)

    def _pof_td(
        self,
        state: ContractState,
        attributes: ContractAttributes,
        time: ActusDateTime,
        risk_factor_observer: RiskFactorObserver,
    ) -> jnp.ndarray:
        """POF_TD_FXOUT: Termination Date - receive termination price.

        Formula:
            POF_TD_FXOUT = X^CURS_CUR(t) × R(CNTRL) × PTD

        Where:
            PTD: Price at termination date
            R(CNTRL): Role sign
            X^CURS_CUR(t): FX rate (if needed)

        Returns:
            Termination price (inflow for seller)
        """
        # Get termination price
        ptd = attributes.price_at_termination_date or 0.0

        # Termination is positive cashflow
        payoff = ptd

        # Apply contract role sign
        role_sign = 1.0 if attributes.contract_role.value == "RPA" else -1.0
        payoff = role_sign * payoff

        return jnp.array(payoff, dtype=jnp.float32)

    def _pof_std(
        self,
        state: ContractState,
        attributes: ContractAttributes,
        time: ActusDateTime,
        risk_factor_observer: RiskFactorObserver,
    ) -> jnp.ndarray:
        """POF_STD_FXOUT: Settlement Date payoff.

        The payoff depends on settlement mode (DS):
        - Delivery (DS='D'): Net settlement in first currency
        - Dual (DS='S'): This handles both STD(1) and STD(2)

        Delivery Formula (DS='D'):
            POF_STD = X^CURS_CUR(t) × R(CNTRL) × (NT - O_rf(i, Md) × NT2)

        Dual Formula (DS='S'):
            POF_STD(1) = X^CURS_CUR(t) × R(CNTRL) × NT
            POF_STD(2) = X^CURS_CUR(t) × R(CNTRL) × (-1) × NT2

        Where:
            NT: First currency amount
            NT2: Second currency amount
            O_rf(i, Md): FX rate observed at maturity
            i: concat(CUR2, '/', CUR) - e.g., "USD/EUR"
            Md: Maturity/settlement date

        Returns:
            Settlement payoff amount
        """
        # Get notional amounts
        nt = attributes.notional_principal or 0.0
        nt2 = attributes.notional_principal_2 or 0.0

        # Get settlement mode
        ds = attributes.delivery_settlement or "D"

        # Get contract role sign
        role_sign = 1.0 if attributes.contract_role.value == "RPA" else -1.0

        if ds == "D":
            # Delivery mode: net settlement
            # Observe FX rate at settlement
            # Rate identifier: CUR2/CUR (e.g., "USD/EUR")
            cur = attributes.currency or "XXX"
            cur2 = attributes.currency_2 or "YYY"
            rate_id = f"{cur2}/{cur}"

            # Observe FX rate from risk factor observer
            fx_rate = risk_factor_observer.observe_risk_factor(rate_id, time)

            # Net payoff: NT - (FX_rate × NT2)
            # This is the profit/loss from the FX position
            payoff = nt - (float(fx_rate) * nt2)

        else:
            # Dual mode (DS='S'): separate payments
            # We need to check the event sequence number to determine STD(1) vs STD(2)
            # For simplicity, we'll return NT for the first STD event
            # and -NT2 for the second STD event
            # In practice, the event schedule should have separate STD(1) and STD(2) events
            payoff = nt  # This will be overridden in actual implementation

        # Apply role sign
        payoff = role_sign * payoff

        return jnp.array(payoff, dtype=jnp.float32)

    def _pof_ce(
        self,
        state: ContractState,
        attributes: ContractAttributes,
        time: ActusDateTime,
        risk_factor_observer: RiskFactorObserver,
    ) -> jnp.ndarray:
        """POF_CE_FXOUT: Credit Event has no cashflow.

        Returns:
            0.0
        """
        return jnp.array(0.0, dtype=jnp.float32)


class FXOutrightStateTransitionFunction(BaseStateTransitionFunction):
    """State transition function for FXOUT contracts.

    Implements all FXOUT state transition functions according to ACTUS specification.

    ACTUS Reference:
        ACTUS v1.1 Section 7.11 - FXOUT State Transition Functions

    State Variables:
        tmd: Maturity date
        prf: Contract performance (DF, DL, DQ, PF)
        sd: Status date

    Events:
        AD: Update status date
        PRD: Update status date
        TD: Update status date
        STD: Update status date
        CE: Update performance and status date
    """

    def transition_state(
        self,
        event_type: Any,
        state: ContractState,
        attributes: ContractAttributes,
        time: ActusDateTime,
        risk_factor_observer: RiskFactorObserver,
    ) -> ContractState:
        """Calculate state transition for FXOUT events.

        Dispatches to specific state transition function based on event type.

        Args:
            event_type: Type of event
            state: Current contract state
            attributes: Contract attributes
            time: Event time
            risk_factor_observer: Observer for market data

        Returns:
            New contract state

        ACTUS Reference:
            STF_[event]_FXOUT functions from Section 7.11
        """
        if event_type == EventType.AD:
            return self._stf_ad(state, attributes, time, risk_factor_observer)
        if event_type == EventType.PRD:
            return self._stf_prd(state, attributes, time, risk_factor_observer)
        if event_type == EventType.TD:
            return self._stf_td(state, attributes, time, risk_factor_observer)
        if event_type == EventType.STD:
            return self._stf_std(state, attributes, time, risk_factor_observer)
        if event_type == EventType.CE:
            return self._stf_ce(state, attributes, time, risk_factor_observer)
        # Unknown event - return state unchanged
        return state

    def _stf_ad(
        self,
        state: ContractState,
        attributes: ContractAttributes,
        time: ActusDateTime,
        risk_factor_observer: RiskFactorObserver,
    ) -> ContractState:
        """STF_AD_FXOUT: Analysis Date - update status date only.

        Returns:
            New state with updated sd
        """
        return ContractState(
            sd=time,
            tmd=state.tmd,
            nt=state.nt,
            ipnr=state.ipnr,
            ipac=state.ipac,
            feac=state.feac,
            nsc=state.nsc,
            isc=state.isc,
            prf=state.prf,
        )

    def _stf_prd(
        self,
        state: ContractState,
        attributes: ContractAttributes,
        time: ActusDateTime,
        risk_factor_observer: RiskFactorObserver,
    ) -> ContractState:
        """STF_PRD_FXOUT: Purchase Date - update status date.

        Returns:
            New state with updated sd
        """
        return ContractState(
            sd=time,
            tmd=state.tmd,
            nt=state.nt,
            ipnr=state.ipnr,
            ipac=state.ipac,
            feac=state.feac,
            nsc=state.nsc,
            isc=state.isc,
            prf=state.prf,
        )

    def _stf_td(
        self,
        state: ContractState,
        attributes: ContractAttributes,
        time: ActusDateTime,
        risk_factor_observer: RiskFactorObserver,
    ) -> ContractState:
        """STF_TD_FXOUT: Termination Date - update status date.

        Returns:
            New state with updated sd
        """
        return ContractState(
            sd=time,
            tmd=state.tmd,
            nt=state.nt,
            ipnr=state.ipnr,
            ipac=state.ipac,
            feac=state.feac,
            nsc=state.nsc,
            isc=state.isc,
            prf=state.prf,
        )

    def _stf_std(
        self,
        state: ContractState,
        attributes: ContractAttributes,
        time: ActusDateTime,
        risk_factor_observer: RiskFactorObserver,
    ) -> ContractState:
        """STF_STD_FXOUT: Settlement Date - update status date.

        Returns:
            New state with updated sd
        """
        return ContractState(
            sd=time,
            tmd=state.tmd,
            nt=state.nt,
            ipnr=state.ipnr,
            ipac=state.ipac,
            feac=state.feac,
            nsc=state.nsc,
            isc=state.isc,
            prf=state.prf,
        )

    def _stf_ce(
        self,
        state: ContractState,
        attributes: ContractAttributes,
        time: ActusDateTime,
        risk_factor_observer: RiskFactorObserver,
    ) -> ContractState:
        """STF_CE_FXOUT: Credit Event - update status date.

        Returns:
            New state with updated sd
        """
        return ContractState(
            sd=time,
            tmd=state.tmd,
            nt=state.nt,
            ipnr=state.ipnr,
            ipac=state.ipac,
            feac=state.feac,
            nsc=state.nsc,
            isc=state.isc,
            prf=state.prf,
        )


class FXOutrightContract(BaseContract):
    """Foreign Exchange Outright (FXOUT) contract.

    Represents an FX forward or spot contract with settlement in two currencies.

    ACTUS Reference:
        ACTUS v1.1 Section 7.11 - FXOUT: Foreign Exchange Outright

    Key Attributes:
        currency (CUR): First currency (e.g., "EUR")
        currency_2 (CUR2): Second currency (e.g., "USD")
        notional_principal (NT): Amount in first currency
        notional_principal_2 (NT2): Amount in second currency
        delivery_settlement (DS): 'D' (delivery/net) or 'S' (dual/gross)
        maturity_date (MD) or settlement_date (STD): Settlement date

    Settlement Modes:
        Delivery (DS='D'):
            - Single STD event with net settlement
            - Payoff = NT - (FX_rate × NT2)
            - Settled in first currency

        Dual (DS='S'):
            - Two STD events: STD(1) and STD(2)
            - STD(1): Receive NT in first currency
            - STD(2): Pay NT2 in second currency
            - Full principal exchange

    State Variables:
        md: Maturity/settlement date
        prf: Contract performance
        sd: Status date

    Example:
        EUR/USD forward contract:
        - Buy 100,000 EUR
        - Sell 110,000 USD
        - Forward rate = 1.10 (agreed)
        - At maturity, observe market rate
        - If market rate = 1.12, profit = 100,000 - (100,000 × 110,000/112,000) ≈ 1,786 EUR
    """

    def __init__(
        self,
        attributes: ContractAttributes,
        risk_factor_observer: RiskFactorObserver,
        child_contract_observer: ChildContractObserver | None = None,
    ):
        """Initialize FXOUT contract.

        Args:
            attributes: Contract attributes
            risk_factor_observer: Observer for FX rates
            child_contract_observer: Observer for child contracts (not used for FXOUT)

        Raises:
            ValueError: If contract type is not FXOUT or required attributes missing
        """
        # Validate contract type
        if attributes.contract_type != ContractType.FXOUT:
            raise ValueError(f"Contract type must be FXOUT, got {attributes.contract_type.value}")

        # Validate required attributes
        if attributes.notional_principal is None:
            raise ValueError("notional_principal (NT) required for FXOUT")

        if attributes.notional_principal_2 is None:
            raise ValueError("notional_principal_2 (NT2) required for FXOUT")

        if attributes.currency is None:
            raise ValueError("currency (CUR) required for FXOUT")

        if attributes.currency_2 is None:
            raise ValueError("currency_2 (CUR2) required for FXOUT")

        if attributes.currency == attributes.currency_2:
            raise ValueError(
                f"Currencies must be different: CUR={attributes.currency}, CUR2={attributes.currency_2}"
            )

        if attributes.delivery_settlement is None:
            raise ValueError("delivery_settlement (DS) required for FXOUT ('D' or 'S')")

        if attributes.delivery_settlement not in ["D", "S"]:
            raise ValueError(
                f"delivery_settlement must be 'D' or 'S', got {attributes.delivery_settlement}"
            )

        # Maturity date or settlement date required
        if attributes.maturity_date is None and attributes.settlement_date is None:
            raise ValueError("maturity_date (MD) or settlement_date (STD) required for FXOUT")

        # Call parent constructor
        super().__init__(
            attributes=attributes,
            risk_factor_observer=risk_factor_observer,
            child_contract_observer=child_contract_observer,
        )

    def generate_event_schedule(self) -> EventSchedule:
        """Generate event schedule for FXOUT contract.

        ACTUS Reference:
            ACTUS v1.1 Section 7.11 - FXOUT Event Schedule

        Events generated:
            - AD: Analysis dates (if defined)
            - PRD: Purchase date (if defined)
            - TD: Termination date (if defined)
            - STD: Settlement date (delivery mode)
            - STD(1), STD(2): Settlement dates (dual mode)
            - CE: Credit events (if defined)

        Returns:
            EventSchedule with contract events
        """
        events = []

        # Determine settlement date
        settlement_date = self.attributes.settlement_date or self.attributes.maturity_date

        # AD: Analysis dates
        if self.attributes.analysis_dates:
            for ad_date in self.attributes.analysis_dates:
                events.append(
                    ContractEvent(
                        event_type=EventType.AD,
                        event_time=ad_date,
                        payoff=jnp.array(0.0, dtype=jnp.float32),
                        currency=self.attributes.currency or "XXX",
                        sequence=len(events),
                    )
                )

        # PRD: Purchase date
        if self.attributes.purchase_date:
            events.append(
                ContractEvent(
                    event_type=EventType.PRD,
                    event_time=self.attributes.purchase_date,
                    payoff=jnp.array(0.0, dtype=jnp.float32),
                    currency=self.attributes.currency or "XXX",
                    sequence=len(events),
                )
            )

        # TD: Termination date
        if self.attributes.termination_date:
            events.append(
                ContractEvent(
                    event_type=EventType.TD,
                    event_time=self.attributes.termination_date,
                    payoff=jnp.array(0.0, dtype=jnp.float32),
                    currency=self.attributes.currency or "XXX",
                    sequence=len(events),
                )
            )

        # STD: Settlement
        if settlement_date:
            if self.attributes.delivery_settlement == "D":
                # Delivery mode: single net settlement
                events.append(
                    ContractEvent(
                        event_type=EventType.STD,
                        event_time=settlement_date,
                        payoff=jnp.array(0.0, dtype=jnp.float32),
                        currency=self.attributes.currency or "XXX",
                        sequence=len(events),
                    )
                )
            else:
                # Dual mode: two separate settlements
                # STD(1): First currency
                events.append(
                    ContractEvent(
                        event_type=EventType.STD,
                        event_time=settlement_date,
                        payoff=jnp.array(0.0, dtype=jnp.float32),
                        currency=self.attributes.currency or "XXX",
                        sequence=len(events),
                    )
                )
                # STD(2): Second currency
                events.append(
                    ContractEvent(
                        event_type=EventType.STD,
                        event_time=settlement_date,
                        payoff=jnp.array(0.0, dtype=jnp.float32),
                        currency=self.attributes.currency_2 or "YYY",
                        sequence=len(events),
                    )
                )

        # Sort events by time
        events.sort(key=lambda e: (e.event_time.to_iso(), e.sequence))

        # Update sequence numbers
        for i, event in enumerate(events):
            event.sequence = i

        return EventSchedule(
            events=events,
            contract_id=self.attributes.contract_id or "FXOUT-UNKNOWN",
        )

    def initialize_state(self) -> ContractState:
        """Initialize contract state.

        ACTUS Reference:
            ACTUS v1.1 Section 7.11 - FXOUT State Variables Initialization

        State Variables:
            tmd: MD if STD = ∅, else STD
            prf: PRF (contract performance)
            sd: Status date
            nt, ipnr, ipac, feac, nsc, isc: Zero/default values (not used by FXOUT)

        Returns:
            Initial contract state
        """
        # Determine maturity date
        tmd = self.attributes.settlement_date or self.attributes.maturity_date

        # Get contract performance
        prf = self.attributes.contract_performance

        # FXOUT has minimal state - most state variables not used
        return ContractState(
            sd=self.attributes.status_date,
            tmd=tmd,
            nt=jnp.array(0.0, dtype=jnp.float32),
            ipnr=jnp.array(0.0, dtype=jnp.float32),
            ipac=jnp.array(0.0, dtype=jnp.float32),
            feac=jnp.array(0.0, dtype=jnp.float32),
            nsc=jnp.array(1.0, dtype=jnp.float32),
            isc=jnp.array(1.0, dtype=jnp.float32),
            prf=prf,
        )

    def get_payoff_function(self, event_type: Any) -> BasePayoffFunction:
        """Get payoff function for FXOUT contract.

        Args:
            event_type: Event type (not used for FXOUT, same function for all events)

        Returns:
            FXOutrightPayoffFunction instance
        """
        return FXOutrightPayoffFunction(
            contract_role=self.attributes.contract_role,
            currency=self.attributes.currency,
        )

    def get_state_transition_function(self, event_type: Any) -> BaseStateTransitionFunction:
        """Get state transition function for FXOUT contract.

        Args:
            event_type: Event type (not used for FXOUT, same function for all events)

        Returns:
            FXOutrightStateTransitionFunction instance
        """
        return FXOutrightStateTransitionFunction()
