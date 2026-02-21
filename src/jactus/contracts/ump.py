"""Undefined Maturity Profile (UMP) contract implementation.

This module implements the UMP contract type from the ACTUS specification.
UMP represents a loan with uncertain principal repayment schedule where all
principal changes come from observed events rather than a fixed schedule.

Key Features:
    - All PR/PI events from observer (no fixed schedule)
    - No IP schedule (only IPCI for capitalization)
    - Uncertain cashflows (all principal changes observed)
    - Maturity from last observed event
    - Same state variables as CLM (no prnxt or ipcb)

Example:
    >>> from jactus import UndefinedMaturityProfileContract, ContractAttributes
    >>> from jactus.core.types import ContractType, ContractRole
    >>> from jactus.core.datetime import ActusDateTime
    >>>
    >>> # Create UMP contract (line of credit with uncertain repayments)
    >>> attrs = ContractAttributes(
    ...     contract_id="UMP-001",
    ...     contract_type=ContractType.UMP,
    ...     contract_role=ContractRole.RPA,
    ...     status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
    ...     initial_exchange_date=ActusDateTime(2024, 1, 15, 0, 0, 0),
    ...     currency="USD",
    ...     notional_principal=100000.0,
    ...     nominal_interest_rate=0.06,
    ...     day_count_convention=DayCountConvention.A360,
    ...     # No maturity date - determined from observed events
    ...     # No IP cycle - only IPCI
    ...     ipci_cycle="1Q",  # Quarterly capitalization
    ... )
    >>> contract = UndefinedMaturityProfileContract(attrs)
    >>> # PR events will come from observer observations

References:
    ACTUS v1.1 Section 7.7 - Undefined Maturity Profile (UMP)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import jax.numpy as jnp

from jactus.contracts.base import BaseContract
from jactus.core.states import ContractState
from jactus.core.types import ContractType, EventType
from jactus.functions import BasePayoffFunction, BaseStateTransitionFunction
from jactus.utilities import contract_role_sign, generate_schedule, year_fraction

if TYPE_CHECKING:
    from jactus.core.attributes import ContractAttributes
    from jactus.core.datetime import ActusDateTime
    from jactus.core.events import ContractEvent, EventSchedule
    from jactus.observers.risk_factor import RiskFactorObserver


class UMPPayoffFunction(BasePayoffFunction):
    """Payoff function for UMP contracts.

    UMP payoffs are simpler than amortizing contracts:
    - IED: Disburse principal
    - PR: Return partial principal (from observer)
    - MD: Return remaining principal + accrued interest
    - IPCI: No payoff (capitalization is internal)
    - FP: Pay accrued fees
    - RR/RRF: No payoff

    All principal repayment amounts come from observed events.
    """

    def __init__(self, contract_role, currency, settlement_currency=None):
        super().__init__(
            contract_role=contract_role,
            currency=currency,
            settlement_currency=settlement_currency,
        )

    def calculate_payoff(
        self,
        event_type: EventType,
        state: ContractState,
        attributes: ContractAttributes,
        time: ActusDateTime,
        risk_factor_observer: RiskFactorObserver,
    ) -> jnp.ndarray:
        """Calculate payoff for given event type.

        Args:
            event_type: Type of event
            state: Current contract state
            attributes: Contract attributes
            time: Event time
            risk_factor_observer: Risk factor observer

        Returns:
            Payoff amount as JAX array
        """
        if event_type == EventType.AD:
            return self._pof_ad(state, attributes, time)
        elif event_type == EventType.IED:
            return self._pof_ied(state, attributes, time)
        elif event_type == EventType.PR:
            return self._pof_pr(state, attributes, time)
        elif event_type == EventType.MD:
            return self._pof_md(state, attributes, time)
        elif event_type == EventType.FP:
            return self._pof_fp(state, attributes, time)
        elif event_type == EventType.TD:
            return self._pof_td(state, attributes, time)
        elif event_type == EventType.IPCI:
            return self._pof_ipci(state, attributes, time)
        elif event_type == EventType.RR:
            return self._pof_rr(state, attributes, time)
        elif event_type == EventType.RRF:
            return self._pof_rrf(state, attributes, time)
        else:
            return jnp.array(0.0, dtype=jnp.float32)

    def _pof_ad(
        self, state: ContractState, attrs: ContractAttributes, time: ActusDateTime
    ) -> jnp.ndarray:
        """POF_AD: Analysis Date - no payoff."""
        return jnp.array(0.0, dtype=jnp.float32)

    def _pof_ied(
        self, state: ContractState, attrs: ContractAttributes, time: ActusDateTime
    ) -> jnp.ndarray:
        """POF_IED: Initial Exchange - disburse principal.

        Formula: R(CNTRL) × (-1) × Nsc × NT
        """
        role_sign = contract_role_sign(attrs.contract_role)
        return jnp.array(role_sign * (-1.0), dtype=jnp.float32) * state.nsc * attrs.notional_principal

    def _pof_pr(
        self, state: ContractState, attrs: ContractAttributes, time: ActusDateTime
    ) -> jnp.ndarray:
        """POF_PR: Principal Repayment - return partial principal.

        For UMP, the amount comes from observed events (not a fixed schedule).
        No R(CNTRL) — state variables are signed.
        """
        return state.nsc * jnp.array(0.0, dtype=jnp.float32)

    def _pof_md(
        self, state: ContractState, attrs: ContractAttributes, time: ActusDateTime
    ) -> jnp.ndarray:
        """POF_MD: Maturity - return principal + accrued interest.

        No R(CNTRL) — all signed state variables.
        """
        yf = year_fraction(state.sd, time, attrs.day_count_convention or attrs.day_count_convention)
        accrued = yf * state.ipnr * state.nt
        return state.nsc * (state.nt + state.ipac + accrued)

    def _pof_td(
        self, state: ContractState, attrs: ContractAttributes, time: ActusDateTime
    ) -> jnp.ndarray:
        """POF_TD: Termination - pay termination price + accrued interest.

        Formula: Nsc × (PTD + IPAC + Y × Nrt × NT)
        """
        ptd = attrs.price_at_termination_date or 0.0
        yf = year_fraction(state.sd, time, attrs.day_count_convention or attrs.day_count_convention)
        accrued = yf * state.ipnr * state.nt
        return state.nsc * jnp.array(ptd + float(state.ipac) + float(accrued), dtype=jnp.float32)

    def _pof_fp(
        self, state: ContractState, attrs: ContractAttributes, time: ActusDateTime
    ) -> jnp.ndarray:
        """POF_FP: Fee Payment - pay accrued fees.

        No R(CNTRL) — Feac is a signed state variable.
        """
        role_sign = contract_role_sign(attrs.contract_role)
        yf = year_fraction(state.sd, time, attrs.day_count_convention or attrs.day_count_convention)

        if attrs.fee_rate and attrs.fee_basis:
            accrued_fees = yf * attrs.fee_rate * abs(state.nt)
            return state.nsc * (state.feac + accrued_fees)
        else:
            return role_sign * state.nsc * state.feac

    def _pof_ipci(
        self, state: ContractState, attrs: ContractAttributes, time: ActusDateTime
    ) -> jnp.ndarray:
        """POF_IPCI: Interest Capitalization - no payoff."""
        return jnp.array(0.0, dtype=jnp.float32)

    def _pof_rr(
        self, state: ContractState, attrs: ContractAttributes, time: ActusDateTime
    ) -> jnp.ndarray:
        """POF_RR: Rate Reset - no payoff."""
        return jnp.array(0.0, dtype=jnp.float32)

    def _pof_rrf(
        self, state: ContractState, attrs: ContractAttributes, time: ActusDateTime
    ) -> jnp.ndarray:
        """POF_RRF: Rate Reset Fixing - no payoff."""
        return jnp.array(0.0, dtype=jnp.float32)


class UMPStateTransitionFunction(BaseStateTransitionFunction):
    """State transition function for UMP contracts.

    UMP state transitions are similar to CLM but with all PR from observer:
    - IED: Initialize notional and rate
    - PR: Reduce notional (amount from observer)
    - MD: Zero out all state variables
    - FP: Reset accrued fees
    - IPCI: Capitalize interest into notional
    - RR: Update interest rate
    - RRF: Fix interest rate
    """

    def transition_state(
        self,
        event_type: EventType,
        state: ContractState,
        attributes: ContractAttributes,
        time: ActusDateTime,
        risk_factor_observer: RiskFactorObserver,
    ) -> ContractState:
        """Execute state transition for given event type.

        Args:
            event_type: Type of event
            state: Current state
            attributes: Contract attributes
            time: Event time
            risk_factor_observer: Risk factor observer

        Returns:
            New contract state
        """
        if event_type == EventType.AD:
            return self._stf_ad(state, attributes, time, risk_factor_observer)
        elif event_type == EventType.IED:
            return self._stf_ied(state, attributes, time, risk_factor_observer)
        elif event_type == EventType.PR:
            return self._stf_pr(state, attributes, time, risk_factor_observer)
        elif event_type == EventType.MD:
            return self._stf_md(state, attributes, time, risk_factor_observer)
        elif event_type == EventType.FP:
            return self._stf_fp(state, attributes, time, risk_factor_observer)
        elif event_type == EventType.IPCI:
            return self._stf_ipci(state, attributes, time, risk_factor_observer)
        elif event_type == EventType.TD:
            return self._stf_td(state, attributes, time, risk_factor_observer)
        elif event_type == EventType.RR:
            return self._stf_rr(state, attributes, time, risk_factor_observer)
        elif event_type == EventType.RRF:
            return self._stf_rrf(state, attributes, time, risk_factor_observer)
        elif event_type == EventType.CE:
            return self._stf_ce(state, attributes, time, risk_factor_observer)
        else:
            return state

    def _stf_ad(
        self,
        state: ContractState,
        attrs: ContractAttributes,
        time: ActusDateTime,
        risk_factor_observer: RiskFactorObserver,
    ) -> ContractState:
        """STF_AD: Analysis Date - just update status date."""
        return state.replace(sd=time)

    def _stf_ied(
        self,
        state: ContractState,
        attrs: ContractAttributes,
        time: ActusDateTime,
        risk_factor_observer: RiskFactorObserver,
    ) -> ContractState:
        """STF_IED: Initial Exchange - set up initial state."""
        nt = jnp.array(attrs.notional_principal, dtype=jnp.float32)
        ipnr = jnp.array(attrs.nominal_interest_rate or 0.0, dtype=jnp.float32)

        return state.replace(
            sd=time,
            nt=nt,
            ipnr=ipnr,
            ipac=jnp.array(0.0, dtype=jnp.float32),
            feac=jnp.array(0.0, dtype=jnp.float32),
            nsc=jnp.array(1.0, dtype=jnp.float32),
            isc=jnp.array(1.0, dtype=jnp.float32),
        )

    def _stf_pr(
        self,
        state: ContractState,
        attrs: ContractAttributes,
        time: ActusDateTime,
        risk_factor_observer: RiskFactorObserver,
    ) -> ContractState:
        """STF_PR: Principal Repayment - reduce notional.

        For UMP, all principal repayments come from observed events.
        The amount is obtained from the observer.
        """
        yf = year_fraction(state.sd, time, attrs.day_count_convention or attrs.day_count_convention)

        # Accrue interest
        new_ipac = state.ipac + yf * state.ipnr * state.nt

        # Get principal repayment amount from observer
        # In practice, this would come from observer.observe_event()
        # For now, we use a placeholder that tests will override
        pr_amount = jnp.array(0.0, dtype=jnp.float32)

        # Reduce notional
        new_nt = state.nt - pr_amount

        return state.replace(sd=time, nt=new_nt, ipac=new_ipac)

    def _stf_md(
        self,
        state: ContractState,
        attrs: ContractAttributes,
        time: ActusDateTime,
        risk_factor_observer: RiskFactorObserver,
    ) -> ContractState:
        """STF_MD: Maturity - zero out all state variables."""
        return state.replace(
            sd=time,
            nt=jnp.array(0.0, dtype=jnp.float32),
            ipnr=jnp.array(0.0, dtype=jnp.float32),
            ipac=jnp.array(0.0, dtype=jnp.float32),
            feac=jnp.array(0.0, dtype=jnp.float32),
        )

    def _stf_fp(
        self,
        state: ContractState,
        attrs: ContractAttributes,
        time: ActusDateTime,
        risk_factor_observer: RiskFactorObserver,
    ) -> ContractState:
        """STF_FP: Fee Payment - reset accrued fees."""
        # Reset fees after payment
        return state.replace(sd=time, feac=jnp.array(0.0, dtype=jnp.float32))

    def _stf_ipci(
        self,
        state: ContractState,
        attrs: ContractAttributes,
        time: ActusDateTime,
        risk_factor_observer: RiskFactorObserver,
    ) -> ContractState:
        """STF_IPCI: Interest Capitalization - add accrued interest to notional."""
        yf = year_fraction(state.sd, time, attrs.day_count_convention or attrs.day_count_convention)
        accrued = yf * state.ipnr * state.nt

        # Add accrued interest to notional (no role_sign - nt is already signed)
        new_nt = state.nt + state.ipac + accrued

        return state.replace(sd=time, nt=new_nt, ipac=jnp.array(0.0, dtype=jnp.float32))

    def _stf_rr(
        self,
        state: ContractState,
        attrs: ContractAttributes,
        time: ActusDateTime,
        risk_factor_observer: RiskFactorObserver,
    ) -> ContractState:
        """STF_RR: Rate Reset - update interest rate from observer."""
        # Get new rate from observer using the market object identifier
        identifier = attrs.rate_reset_market_object or "RATE"
        new_rate = risk_factor_observer.observe_risk_factor(identifier, time, state, attrs)

        return state.replace(sd=time, ipnr=new_rate)

    def _stf_rrf(
        self,
        state: ContractState,
        attrs: ContractAttributes,
        time: ActusDateTime,
        risk_factor_observer: RiskFactorObserver,
    ) -> ContractState:
        """STF_RRF: Rate Reset Fixing - fix interest rate."""
        # For UMP, similar to RR but uses fixed rate from attributes
        if attrs.rate_reset_next is not None:
            new_rate = jnp.array(attrs.rate_reset_next, dtype=jnp.float32)
        else:
            new_rate = state.ipnr

        return state.replace(sd=time, ipnr=new_rate)

    def _stf_td(
        self,
        state: ContractState,
        attrs: ContractAttributes,
        time: ActusDateTime,
        risk_factor_observer: RiskFactorObserver,
    ) -> ContractState:
        """STF_TD: Termination - zero out notional/accrued, keep rate."""
        return state.replace(
            sd=time,
            nt=jnp.array(0.0, dtype=jnp.float32),
            ipac=jnp.array(0.0, dtype=jnp.float32),
            feac=jnp.array(0.0, dtype=jnp.float32),
        )

    def _stf_ce(
        self,
        state: ContractState,
        attrs: ContractAttributes,
        time: ActusDateTime,
        risk_factor_observer: RiskFactorObserver,
    ) -> ContractState:
        """STF_CE: Credit Event - not yet implemented."""
        return state.replace(sd=time)


class UndefinedMaturityProfileContract(BaseContract):
    """Undefined Maturity Profile (UMP) contract.

    UMP represents a loan with uncertain principal repayment schedule.
    All principal changes (PR/PI events) come from observed events rather
    than a fixed schedule. This is useful for modeling:
    - Lines of credit with uncertain drawdowns/repayments
    - Loans with irregular principal payments
    - Cashflow profiles determined by external events

    Key Characteristics:
        - All PR events from observer (no fixed schedule)
        - No IP schedule (only IPCI for capitalization)
        - Uncertain cashflows (all principal from observations)
        - Maturity from last observed event
        - Simpler state than LAM (no prnxt or ipcb)

    Event Types:
        - AD: Analysis Date
        - IED: Initial Exchange
        - PR: Principal Repayment (from observer)
        - PI: Principal Increase (from observer)
        - FP: Fee Payment
        - IPCI: Interest Capitalization
        - RR: Rate Reset
        - RRF: Rate Reset Fixing
        - CE: Credit Event
        - MD: Maturity

    State Variables:
        - sd: Status Date
        - tmd: Terminal Maturity Date (from observed events)
        - nt: Notional
        - ipnr: Interest Rate
        - ipac: Accrued Interest
        - feac: Accrued Fees
        - nsc: Notional Scaling
        - isc: Interest Scaling

    Example:
        >>> attrs = ContractAttributes(
        ...     contract_id="UMP-001",
        ...     contract_type=ContractType.UMP,
        ...     contract_role=ContractRole.RPA,
        ...     status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
        ...     initial_exchange_date=ActusDateTime(2024, 1, 15, 0, 0, 0),
        ...     currency="USD",
        ...     notional_principal=100000.0,
        ...     nominal_interest_rate=0.06,
        ...     ipci_cycle="1Q",  # Quarterly capitalization
        ... )
        >>> contract = UndefinedMaturityProfileContract(attrs)

    References:
        ACTUS v1.1 Section 7.7 - Undefined Maturity Profile
    """

    def __init__(
        self,
        attributes: ContractAttributes,
        risk_factor_observer: RiskFactorObserver,
        child_contract_observer: Any | None = None,
    ):
        """Initialize UMP contract.

        Args:
            attributes: Contract attributes
            risk_factor_observer: Risk factor observer for rate updates
            child_contract_observer: Optional child contract observer

        Raises:
            ValueError: If contract_type is not UMP
        """
        # Validate contract type
        if attributes.contract_type != ContractType.UMP:
            raise ValueError(f"Contract type must be UMP, got {attributes.contract_type.value}")

        super().__init__(
            attributes=attributes,
            risk_factor_observer=risk_factor_observer,
            child_contract_observer=child_contract_observer,
        )

    def initialize_state(self) -> ContractState:
        """Initialize UMP state.

        UMP state is simpler than LAM - no prnxt or ipcb states.
        Maturity is determined from observed events.
        """
        return ContractState(
            sd=self.attributes.status_date,
            tmd=self.attributes.maturity_date,  # May be None - determined from observer
            nt=jnp.array(0.0, dtype=jnp.float32),  # Set at IED
            ipnr=jnp.array(0.0, dtype=jnp.float32),  # Set at IED
            ipac=jnp.array(0.0, dtype=jnp.float32),
            feac=jnp.array(0.0, dtype=jnp.float32),
            nsc=jnp.array(1.0, dtype=jnp.float32),
            isc=jnp.array(1.0, dtype=jnp.float32),
        )

    def get_payoff_function(self, event_type: Any) -> UMPPayoffFunction:
        """Get UMP payoff function.

        Args:
            event_type: Event type (unused - same POF for all)

        Returns:
            UMP payoff function instance
        """
        return UMPPayoffFunction(
            contract_role=self.attributes.contract_role,
            currency=self.attributes.currency,
            settlement_currency=self.attributes.settlement_currency,
        )

    def get_state_transition_function(self, event_type: Any) -> UMPStateTransitionFunction:
        """Get UMP state transition function.

        Args:
            event_type: Event type (unused - same STF for all)

        Returns:
            UMP state transition function instance
        """
        return UMPStateTransitionFunction()

    def generate_event_schedule(self) -> EventSchedule:
        """Generate UMP event schedule.

        UMP schedule includes:
        - AD: Analysis date (if provided)
        - IED: Initial exchange
        - IPCI: Interest capitalization (from interest_payment_cycle)
        - RR: Rate reset (if RR cycle provided)
        - RRF: Rate reset fixing (if RRF cycle provided)
        - FP: Fee payment (if FP cycle provided)
        - TD: Termination (if termination_date is set)
        - MD: Maturity (from observed events or attributes)

        Note: PR/PI events come from observer, not from schedule.

        Returns:
            Event schedule with all scheduled events
        """
        from jactus.core.events import ContractEvent, EventSchedule

        attributes = self.attributes
        events: list[ContractEvent] = []
        ied = attributes.initial_exchange_date

        # AD: Analysis Date (optional)
        if attributes.status_date:
            events.append(
                ContractEvent(
                    event_type=EventType.AD,
                    event_time=attributes.status_date,
                    payoff=jnp.array(0.0, dtype=jnp.float32),
                    currency=attributes.currency or "XXX",
                )
            )

        # IED: Initial Exchange Date (required)
        if ied and attributes.status_date < ied:
            events.append(
                ContractEvent(
                    event_type=EventType.IED,
                    event_time=ied,
                    payoff=jnp.array(0.0, dtype=jnp.float32),
                    currency=attributes.currency or "XXX",
                )
            )

        # Determine end date for periodic schedules
        # TD takes precedence over MD if set
        td = attributes.termination_date
        md = attributes.maturity_date
        end_date = td or md

        if not end_date or not ied:
            # Without any end date, only return AD + IED
            events = sorted(events, key=lambda e: e.event_time)
            return EventSchedule(events=tuple(events), contract_id=attributes.contract_id)

        # IPCI: Interest Capitalization from interest_payment_cycle
        # For UMP, all interest payment dates are IPCI events (no regular IP)
        if attributes.interest_payment_cycle:
            ipci_anchor = attributes.interest_payment_anchor or ied
            ipci_end = attributes.interest_capitalization_end_date or end_date
            ipci_schedule = generate_schedule(
                start=ipci_anchor,
                cycle=attributes.interest_payment_cycle,
                end=ipci_end,
            )
            for ipci_time in ipci_schedule:
                if ied < ipci_time < end_date and ipci_time > attributes.status_date:
                    events.append(
                        ContractEvent(
                            event_type=EventType.IPCI,
                            event_time=ipci_time,
                            payoff=jnp.array(0.0, dtype=jnp.float32),
                            currency=attributes.currency or "XXX",
                        )
                    )

        # RR: Rate Reset (optional)
        if attributes.rate_reset_cycle and attributes.rate_reset_anchor:
            rr_schedule = generate_schedule(
                start=attributes.rate_reset_anchor,
                cycle=attributes.rate_reset_cycle,
                end=end_date,
            )
            for rr_time in rr_schedule:
                if ied < rr_time < end_date:
                    events.append(
                        ContractEvent(
                            event_type=EventType.RR,
                            event_time=rr_time,
                            payoff=jnp.array(0.0, dtype=jnp.float32),
                            currency=attributes.currency or "XXX",
                        )
                    )

        # FP: Fee Payment (optional)
        if attributes.fee_payment_cycle and attributes.fee_payment_anchor:
            fp_schedule = generate_schedule(
                start=attributes.fee_payment_anchor,
                cycle=attributes.fee_payment_cycle,
                end=end_date,
            )
            for fp_time in fp_schedule:
                if ied < fp_time <= end_date:
                    events.append(
                        ContractEvent(
                            event_type=EventType.FP,
                            event_time=fp_time,
                            payoff=jnp.array(0.0, dtype=jnp.float32),
                            currency=attributes.currency or "XXX",
                        )
                    )

        # TD: Termination Date (if set)
        # UMP has no MD event — maturity is uncertain and determined by observed events.
        # The maturity_date field is only used as upper bound for schedule generation.
        if td:
            events.append(
                ContractEvent(
                    event_type=EventType.TD,
                    event_time=td,
                    payoff=jnp.array(0.0, dtype=jnp.float32),
                    currency=attributes.currency or "XXX",
                )
            )

        # Sort events by time
        events = sorted(events, key=lambda e: e.event_time)

        return EventSchedule(events=tuple(events), contract_id=attributes.contract_id)
