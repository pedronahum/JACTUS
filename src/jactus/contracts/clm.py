"""Call Money (CLM) contract implementation.

This module implements the CLM contract type - an on-demand loan where the maturity
is not fixed at inception but determined by observed events (typically a call event
from the lender or repayment from the borrower).

ACTUS Reference:
    ACTUS v1.1 Section 7.6 - CLM: Call Money

Key Features:
    - No fixed maturity date at inception
    - Maturity determined from observed events
    - Single interest payment at termination
    - Optional periodic interest capitalization (IPCI)
    - Principal repaid when called or at observed maturity
    - Simpler than PAM - no regular IP schedule

Typical Use Cases:
    - Lines of credit
    - Overnight/call loans
    - Demand deposits
    - Flexible repayment loans

Example:
    >>> from jactus.contracts import create_contract
    >>> from jactus.core import ContractAttributes, ContractType, ContractRole
    >>> from jactus.core import ActusDateTime, DayCountConvention
    >>> from jactus.observers import ConstantRiskFactorObserver
    >>>
    >>> attrs = ContractAttributes(
    ...     contract_id="LOC-001",
    ...     contract_type=ContractType.CLM,
    ...     contract_role=ContractRole.RPA,
    ...     status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
    ...     initial_exchange_date=ActusDateTime(2024, 1, 15, 0, 0, 0),
    ...     # No maturity_date - determined dynamically
    ...     currency="USD",
    ...     notional_principal=50000.0,
    ...     nominal_interest_rate=0.08,
    ...     day_count_convention=DayCountConvention.A360,
    ... )
    >>>
    >>> rf_obs = ConstantRiskFactorObserver(constant_value=0.08)
    >>> contract = create_contract(attrs, rf_obs)
    >>> result = contract.simulate()
"""

from typing import Any

import jax.numpy as jnp

from jactus.contracts.base import BaseContract, SimulationHistory
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
from jactus.observers import RiskFactorObserver
from jactus.core.types import BusinessDayConvention, Calendar, EndOfMonthConvention
from jactus.utilities import contract_role_sign, generate_schedule, year_fraction


class CLMPayoffFunction(BasePayoffFunction):
    """Payoff function for CLM contracts.

    CLM payoff functions are similar to PAM but simpler:
    - No regular IP events (only at maturity)
    - Maturity is dynamic (from observed events)
    - IPCI events capitalize interest periodically

    ACTUS Reference:
        ACTUS v1.1 Section 7.6 - CLM Payoff Functions

    Events:
        AD: Analysis Date (0.0)
        IED: Initial Exchange Date (disburse principal)
        MD: Maturity Date (return principal + accrued - dynamic)
        PR: Principal Repayment (from observer)
        FP: Fee Payment
        IP: Interest Payment (single event at maturity)
        IPCI: Interest Capitalization
        RR: Rate Reset
        RRF: Rate Reset Fixing
        CE: Credit Event
    """

    def __init__(self, contract_role, currency, settlement_currency=None):
        """Initialize CLM payoff function.

        Args:
            contract_role: Contract role (RPA or RPL)
            currency: Contract currency
            settlement_currency: Optional settlement currency
        """
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
            Payoff amount (JAX array)
        """
        if event_type == EventType.AD:
            return self._pof_ad(state, attributes, time)
        if event_type == EventType.IED:
            return self._pof_ied(state, attributes, time)
        if event_type == EventType.PR:
            return self._pof_pr(state, attributes, time)
        if event_type == EventType.MD:
            return self._pof_md(state, attributes, time)
        if event_type == EventType.FP:
            return self._pof_fp(state, attributes, time)
        if event_type == EventType.IP:
            return self._pof_ip(state, attributes, time)
        if event_type == EventType.IPCI:
            return self._pof_ipci(state, attributes, time)
        if event_type == EventType.RR:
            return self._pof_rr(state, attributes, time)
        if event_type == EventType.RRF:
            return self._pof_rrf(state, attributes, time)
        if event_type == EventType.CE:
            return self._pof_ce(state, attributes, time)
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

        Formula: R(CNTRL) × (-1) × (NT + PDIED)
        """
        role_sign = contract_role_sign(attrs.contract_role)
        nt = attrs.notional_principal or 0.0
        pdied = attrs.premium_discount_at_ied or 0.0
        return jnp.array(role_sign * (-1) * (nt + pdied), dtype=jnp.float32)

    def _pof_pr(
        self, state: ContractState, attrs: ContractAttributes, time: ActusDateTime
    ) -> jnp.ndarray:
        """POF_PR: Principal Repayment - return partial principal.

        For CLM, principal repayments can occur based on observed events.
        No role_sign needed — state.nt is already signed.
        """
        return state.nsc * state.nt

    def _pof_md(
        self, state: ContractState, attrs: ContractAttributes, time: ActusDateTime
    ) -> jnp.ndarray:
        """POF_MD: Maturity - return principal.

        Interest is paid separately by the IP event at maturity.
        No role_sign needed — state.nt is already signed.
        """
        return state.nsc * state.nt

    def _pof_fp(
        self, state: ContractState, attrs: ContractAttributes, time: ActusDateTime
    ) -> jnp.ndarray:
        """POF_FP: Fee Payment - pay accrued fees."""
        return state.feac

    def _pof_ip(
        self, state: ContractState, attrs: ContractAttributes, time: ActusDateTime
    ) -> jnp.ndarray:
        """POF_IP: Interest Payment - pay accrued interest.

        For CLM, this typically only occurs at maturity.
        No role_sign needed — state variables are already signed.
        """
        yf = year_fraction(state.sd, time, attrs.day_count_convention or attrs.day_count_convention)
        accrued = yf * state.ipnr * state.nt
        return state.isc * (state.ipac + accrued)

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

    def _pof_ce(
        self, state: ContractState, attrs: ContractAttributes, time: ActusDateTime
    ) -> jnp.ndarray:
        """POF_CE: Credit Event - not yet implemented."""
        return jnp.array(0.0, dtype=jnp.float32)


class CLMStateTransitionFunction(BaseStateTransitionFunction):
    """State transition function for CLM contracts.

    CLM state transitions are similar to PAM but without IPCB or Prnxt states.

    ACTUS Reference:
        ACTUS v1.1 Section 7.6 - CLM State Transition Functions
    """

    def transition_state(
        self,
        event_type: EventType,
        state: ContractState,
        attributes: ContractAttributes,
        time: ActusDateTime,
        risk_factor_observer: RiskFactorObserver,
    ) -> ContractState:
        """Transition state for given event type.

        Args:
            event_type: Type of event
            state: Current contract state
            attributes: Contract attributes
            time: Event time
            risk_factor_observer: Risk factor observer

        Returns:
            New contract state
        """
        if event_type == EventType.AD:
            return self._stf_ad(state, attributes, time, risk_factor_observer)
        if event_type == EventType.IED:
            return self._stf_ied(state, attributes, time, risk_factor_observer)
        if event_type == EventType.PR:
            return self._stf_pr(state, attributes, time, risk_factor_observer)
        if event_type == EventType.MD:
            return self._stf_md(state, attributes, time, risk_factor_observer)
        if event_type == EventType.FP:
            return self._stf_fp(state, attributes, time, risk_factor_observer)
        if event_type == EventType.IP:
            return self._stf_ip(state, attributes, time, risk_factor_observer)
        if event_type == EventType.IPCI:
            return self._stf_ipci(state, attributes, time, risk_factor_observer)
        if event_type == EventType.RR:
            return self._stf_rr(state, attributes, time, risk_factor_observer)
        if event_type == EventType.RRF:
            return self._stf_rrf(state, attributes, time, risk_factor_observer)
        if event_type == EventType.CE:
            return self._stf_ce(state, attributes, time, risk_factor_observer)
        return state

    def _stf_ad(
        self,
        state: ContractState,
        attrs: ContractAttributes,
        time: ActusDateTime,
        risk_factor_observer: RiskFactorObserver,
    ) -> ContractState:
        """STF_AD: Analysis Date - update status date only."""
        return state.replace(sd=time)

    def _stf_ied(
        self,
        state: ContractState,
        attrs: ContractAttributes,
        time: ActusDateTime,
        risk_factor_observer: RiskFactorObserver,
    ) -> ContractState:
        """STF_IED: Initial Exchange - initialize all state variables."""
        role_sign = contract_role_sign(attrs.contract_role)

        return state.replace(
            sd=time,
            nt=role_sign * jnp.array(attrs.notional_principal, dtype=jnp.float32),
            ipnr=jnp.array(attrs.nominal_interest_rate or 0.0, dtype=jnp.float32),
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

        For CLM, principal repayments reduce the notional.
        The amount comes from observed events.
        """
        yf = year_fraction(state.sd, time, attrs.day_count_convention or attrs.day_count_convention)

        # Accrue interest
        new_ipac = state.ipac + yf * state.ipnr * state.nt

        # For simplicity, assume full repayment
        # In practice, partial amounts would come from observer
        new_nt = jnp.array(0.0, dtype=jnp.float32)

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

    def _stf_ip(
        self,
        state: ContractState,
        attrs: ContractAttributes,
        time: ActusDateTime,
        risk_factor_observer: RiskFactorObserver,
    ) -> ContractState:
        """STF_IP: Interest Payment - reset accrued interest."""
        # Reset interest after payment
        return state.replace(sd=time, ipac=jnp.array(0.0, dtype=jnp.float32))

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
        observation_time: ActusDateTime | None = None,
    ) -> ContractState:
        """STF_RR: Rate Reset - accrue interest, then update rate from observer."""
        # Accrue interest up to this point
        yf = year_fraction(state.sd, time, attrs.day_count_convention or attrs.day_count_convention)
        new_ipac = state.ipac + yf * state.ipnr * state.nt

        # Get new rate from observer (use observation_time for BDC-shifted RR events)
        identifier = attrs.rate_reset_market_object or "RATE"
        obs_time = observation_time or time
        observed = risk_factor_observer.observe_risk_factor(identifier, obs_time, state, attrs)

        # Apply rate multiplier and spread
        multiplier = attrs.rate_reset_multiplier if attrs.rate_reset_multiplier is not None else 1.0
        spread = attrs.rate_reset_spread if attrs.rate_reset_spread is not None else 0.0
        new_rate = multiplier * observed + spread

        # Apply floor/cap
        if attrs.rate_reset_floor is not None:
            new_rate = jnp.maximum(new_rate, jnp.array(attrs.rate_reset_floor, dtype=jnp.float32))
        if attrs.rate_reset_cap is not None:
            new_rate = jnp.minimum(new_rate, jnp.array(attrs.rate_reset_cap, dtype=jnp.float32))

        return state.replace(sd=time, ipac=new_ipac, ipnr=new_rate)

    def _stf_rrf(
        self,
        state: ContractState,
        attrs: ContractAttributes,
        time: ActusDateTime,
        risk_factor_observer: RiskFactorObserver,
    ) -> ContractState:
        """STF_RRF: Rate Reset Fixing - fix interest rate."""
        # For CLM, similar to RR
        if attrs.rate_reset_next is not None:
            new_rate = jnp.array(attrs.rate_reset_next, dtype=jnp.float32)
        else:
            new_rate = state.ipnr

        return state.replace(sd=time, ipnr=new_rate)

    def _stf_ce(
        self,
        state: ContractState,
        attrs: ContractAttributes,
        time: ActusDateTime,
        risk_factor_observer: RiskFactorObserver,
    ) -> ContractState:
        """STF_CE: Credit Event - not yet implemented."""
        return state.replace(sd=time)


class CallMoneyContract(BaseContract):
    """CLM (Call Money) contract implementation.

    CLM is an on-demand loan where maturity is determined by observed events
    rather than fixed at inception. Common for lines of credit and demand loans.

    ACTUS Reference:
        ACTUS v1.1 Section 7.6 - CLM: Call Money
    """

    def __init__(
        self,
        attributes: ContractAttributes,
        risk_factor_observer: RiskFactorObserver,
        child_contract_observer: Any | None = None,
    ):
        """Initialize CLM contract.

        Args:
            attributes: Contract attributes
            risk_factor_observer: Risk factor observer for rate updates
            child_contract_observer: Optional child contract observer

        Raises:
            ValueError: If contract_type is not CLM
        """
        if attributes.contract_type != ContractType.CLM:
            raise ValueError(f"Contract type must be CLM, got {attributes.contract_type.value}")

        super().__init__(
            attributes=attributes,
            risk_factor_observer=risk_factor_observer,
            child_contract_observer=child_contract_observer,
        )

    def initialize_state(self) -> ContractState:
        """Initialize CLM contract state.

        CLM state is simpler than LAM - no prnxt or ipcb states.
        When status_date >= IED, the IED event is skipped so state must be
        initialized from contract attributes directly.

        Returns:
            Initial contract state
        """
        attrs = self.attributes
        ied = attrs.initial_exchange_date

        # When SD >= IED, the IED event won't fire, so pre-initialize
        if ied and attrs.status_date >= ied:
            role_sign = contract_role_sign(attrs.contract_role)
            nt = role_sign * jnp.array(attrs.notional_principal or 0.0, dtype=jnp.float32)
            ipnr = jnp.array(attrs.nominal_interest_rate or 0.0, dtype=jnp.float32)
            ipac = jnp.array(attrs.accrued_interest or 0.0, dtype=jnp.float32)
        else:
            nt = jnp.array(0.0, dtype=jnp.float32)
            ipnr = jnp.array(0.0, dtype=jnp.float32)
            ipac = jnp.array(0.0, dtype=jnp.float32)

        return ContractState(
            sd=attrs.status_date,
            tmd=attrs.maturity_date,
            nt=nt,
            ipnr=ipnr,
            ipac=ipac,
            feac=jnp.array(0.0, dtype=jnp.float32),
            nsc=jnp.array(1.0, dtype=jnp.float32),
            isc=jnp.array(1.0, dtype=jnp.float32),
        )

    def get_payoff_function(self, event_type: Any) -> CLMPayoffFunction:
        """Get CLM payoff function.

        Args:
            event_type: Type of event (not used, all events use same POF)

        Returns:
            CLM payoff function instance
        """
        return CLMPayoffFunction(
            contract_role=self.attributes.contract_role,
            currency=self.attributes.currency,
            settlement_currency=None,
        )

    def get_state_transition_function(self, event_type: Any) -> CLMStateTransitionFunction:
        """Get CLM state transition function.

        Args:
            event_type: Type of event (not used, all events use same STF)

        Returns:
            CLM state transition function instance
        """
        return CLMStateTransitionFunction()

    def simulate(
        self,
        risk_factor_observer: RiskFactorObserver | None = None,
        child_contract_observer: Any | None = None,
    ) -> SimulationHistory:
        """Simulate CLM contract with BDC-aware rate observation.

        For SCP/SCF conventions, RR events use the original (unadjusted)
        schedule date for rate observation, not the BDC-shifted event time.
        """
        risk_obs = risk_factor_observer or self.risk_factor_observer
        state = self.initialize_state()
        initial_state = state
        events_with_states = []
        schedule = self.get_events()
        # Get observation date mapping (populated by generate_event_schedule)
        obs_dates = getattr(self, "_rr_observation_dates", {})

        for event in schedule.events:
            stf = self.get_state_transition_function(event.event_type)
            pof = self.get_payoff_function(event.event_type)
            calc_time = event.calculation_time or event.event_time

            payoff = pof.calculate_payoff(
                event_type=event.event_type,
                state=state,
                attributes=self.attributes,
                time=calc_time,
                risk_factor_observer=risk_obs,
            )

            # For RR events, pass observation time if BDC-shifted
            if event.event_type == EventType.RR and event.event_time.to_iso() in obs_dates:
                obs_time = obs_dates[event.event_time.to_iso()]
                state_post = stf._stf_rr(state, self.attributes, calc_time, risk_obs, obs_time)
            else:
                state_post = stf.transition_state(
                    event_type=event.event_type,
                    state=state,
                    attributes=self.attributes,
                    time=calc_time,
                    risk_factor_observer=risk_obs,
                )

            processed_event = ContractEvent(
                event_type=event.event_type,
                event_time=event.event_time,
                payoff=payoff,
                currency=event.currency or self.attributes.currency or "XXX",
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

    def generate_event_schedule(self) -> EventSchedule:
        """Generate complete event schedule for CLM contract.

        CLM schedule is simpler than PAM:
        - No regular IP events (only at maturity if MD is set)
        - Optional IPCI events for interest capitalization
        - Maturity may be dynamic (from observed events)

        Returns:
            EventSchedule with all contract events
        """
        events = []
        attributes = self.attributes
        ied = attributes.initial_exchange_date

        if not ied:
            return EventSchedule(events=[], contract_id=attributes.contract_id)

        sd = attributes.status_date
        bdc = attributes.business_day_convention or BusinessDayConvention.NULL
        cal = attributes.calendar or Calendar.NO_CALENDAR
        eomc = attributes.end_of_month_convention or EndOfMonthConvention.SD

        def _sched(anchor, cycle, end):
            return generate_schedule(
                start=anchor, cycle=cycle, end=end,
                end_of_month_convention=eomc,
                business_day_convention=bdc,
                calendar=cal,
            )

        # AD: Analysis Date
        events.append(
            ContractEvent(
                event_type=EventType.AD,
                event_time=sd,
                payoff=jnp.array(0.0, dtype=jnp.float32),
                currency=attributes.currency or "XXX",
            )
        )

        # IED: Initial Exchange Date (skip when SD >= IED)
        if sd < ied:
            events.append(
                ContractEvent(
                    event_type=EventType.IED,
                    event_time=ied,
                    payoff=jnp.array(0.0, dtype=jnp.float32),
                    currency=attributes.currency or "XXX",
                )
            )

        # IPCI Schedule: Periodic interest capitalization
        if attributes.interest_payment_cycle and attributes.maturity_date:
            ipci_end = attributes.interest_capitalization_end_date or attributes.maturity_date
            ipci_schedule = _sched(
                attributes.interest_payment_anchor or ied,
                attributes.interest_payment_cycle,
                ipci_end,
            )
            for time in ipci_schedule:
                if time > sd and time < attributes.maturity_date:
                    events.append(
                        ContractEvent(
                            event_type=EventType.IPCI,
                            event_time=time,
                            payoff=jnp.array(0.0, dtype=jnp.float32),
                            currency=attributes.currency or "XXX",
                        )
                    )

        # RR Schedule: Rate resets
        # For SCP/SCF conventions, RR events are BDC-adjusted but observation
        # uses the original (unadjusted) schedule date. Store mapping.
        self._rr_observation_dates: dict[str, ActusDateTime] = {}
        if attributes.maturity_date:
            if attributes.rate_reset_cycle:
                # Generate BDC-adjusted schedule for event timing
                rr_adjusted = _sched(
                    attributes.rate_reset_anchor or ied,
                    attributes.rate_reset_cycle,
                    attributes.maturity_date,
                )
                # Generate unadjusted schedule for rate observation
                rr_unadjusted = generate_schedule(
                    start=attributes.rate_reset_anchor or ied,
                    cycle=attributes.rate_reset_cycle,
                    end=attributes.maturity_date,
                    end_of_month_convention=eomc,
                )
                # Map adjusted→unadjusted dates
                for adj, unadj in zip(rr_adjusted, rr_unadjusted):
                    if adj != unadj:
                        self._rr_observation_dates[adj.to_iso()] = unadj

                for time in rr_adjusted:
                    if time > sd and time <= attributes.maturity_date:
                        events.append(
                            ContractEvent(
                                event_type=EventType.RR,
                                event_time=time,
                                payoff=jnp.array(0.0, dtype=jnp.float32),
                                currency=attributes.currency or "XXX",
                            )
                        )
            elif attributes.rate_reset_anchor and attributes.rate_reset_market_object:
                # Single RR at anchor date (no cycle)
                rr_time = attributes.rate_reset_anchor
                if rr_time > sd and rr_time <= attributes.maturity_date:
                    events.append(
                        ContractEvent(
                            event_type=EventType.RR,
                            event_time=rr_time,
                            payoff=jnp.array(0.0, dtype=jnp.float32),
                            currency=attributes.currency or "XXX",
                        )
                    )

        # MD: Maturity Date (if specified)
        if attributes.maturity_date:
            md = attributes.maturity_date
            events.append(
                ContractEvent(
                    event_type=EventType.IP,
                    event_time=md,
                    payoff=jnp.array(0.0, dtype=jnp.float32),
                    currency=attributes.currency or "XXX",
                )
            )
            events.append(
                ContractEvent(
                    event_type=EventType.MD,
                    event_time=md,
                    payoff=jnp.array(0.0, dtype=jnp.float32),
                    currency=attributes.currency or "XXX",
                )
            )

        # Sort events by time
        events.sort(key=lambda e: (e.event_time, e.event_type.value))

        return EventSchedule(events=events, contract_id=attributes.contract_id)
