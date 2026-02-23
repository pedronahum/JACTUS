"""Negative Amortizer (NAM) contract implementation.

This module implements the NAM contract type - an amortizing loan where principal
can increase (negative amortization) when payments are less than accrued interest.
NAM extends the LAM pattern with modified payoff and state transition functions.

ACTUS Reference:
    ACTUS v1.1 Section 7.4 - NAM: Negative Amortizer

Key Features:
    - Negative amortization: Notional can increase if payment < interest
    - Modified PR payoff: Prnxt - accrued interest (can be negative)
    - Modified PR STF: Notional changes by net payment amount
    - IP schedule ends one period before PR schedule
    - Maturity calculation accounts for negative amortization effect
    - Same states as LAM: prnxt, ipcb
    - Same events as LAM

Negative Amortization:
    When the scheduled principal payment (Prnxt) is less than the accrued interest,
    the shortfall is added to the notional principal:

    - If Prnxt > interest: Normal amortization (notional decreases)
    - If Prnxt < interest: Negative amortization (notional increases)
    - If Prnxt = interest: Interest-only payment (notional unchanged)

Example:
    >>> from jactus.contracts import create_contract
    >>> from jactus.core import ContractAttributes, ContractType, ContractRole
    >>> from jactus.core import ActusDateTime, DayCountConvention
    >>> from jactus.observers import ConstantRiskFactorObserver
    >>>
    >>> attrs = ContractAttributes(
    ...     contract_id="NAM-001",
    ...     contract_type=ContractType.NAM,
    ...     contract_role=ContractRole.RPA,
    ...     status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
    ...     initial_exchange_date=ActusDateTime(2024, 1, 15, 0, 0, 0),
    ...     maturity_date=ActusDateTime(2054, 1, 15, 0, 0, 0),
    ...     currency="USD",
    ...     notional_principal=300000.0,
    ...     nominal_interest_rate=0.065,
    ...     day_count_convention=DayCountConvention.A360,
    ...     principal_redemption_cycle="1M",  # Monthly payments
    ...     next_principal_redemption_amount=800.0,  # Low payment → negative amort
    ...     interest_calculation_base="NT"
    ... )
    >>>
    >>> rf_obs = ConstantRiskFactorObserver(constant_value=0.065)
    >>> contract = create_contract(attrs, rf_obs)
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
    DayCountConvention,
    EventSchedule,
    EventType,
)
from jactus.core.types import (
    EVENT_SCHEDULE_PRIORITY,
    BusinessDayConvention,
    Calendar,
    EndOfMonthConvention,
)
from jactus.functions import BasePayoffFunction, BaseStateTransitionFunction
from jactus.observers import RiskFactorObserver
from jactus.utilities import contract_role_sign, generate_schedule, year_fraction


class NAMPayoffFunction(BasePayoffFunction):
    """Payoff function for NAM contracts.

    Implements all NAM payoff functions according to ACTUS specification.
    The key difference from LAM is the PR event payoff, which is net of
    accrued interest and can be negative.

    ACTUS Reference:
        ACTUS v1.1 Section 7.4 - NAM Payoff Functions

    Events:
        Same as LAM, but PR payoff modified:
        POF_PR = R(CNTRL) × Nsc × (Prnxt - Ipac - Y(Sd, t) × Ipnr × Ipcb)
    """

    def _build_dispatch_table(self) -> dict[EventType, Any]:
        """Build event type → handler dispatch table.

        Lambdas normalize varying handler signatures to a uniform
        (state, attributes, time, risk_factor_observer) interface.
        """
        return {
            EventType.AD: lambda s, a, t, r: self._pof_ad(s, a),
            EventType.IED: lambda s, a, t, r: self._pof_ied(s, a),
            EventType.PR: lambda s, a, t, r: self._pof_pr(s, a, t),
            EventType.MD: lambda s, a, t, r: self._pof_md(s, a),
            EventType.PP: self._pof_pp,
            EventType.PY: self._pof_py,
            EventType.FP: self._pof_fp,
            EventType.PRD: lambda s, a, t, r: self._pof_prd(s, a, t),
            EventType.TD: lambda s, a, t, r: self._pof_td(s, a, t),
            EventType.IP: lambda s, a, t, r: self._pof_ip(s, a, t),
            EventType.IPCI: lambda s, a, t, r: self._pof_ipci(s, a),
            EventType.IPCB: lambda s, a, t, r: self._pof_ipcb(s, a),
            EventType.RR: lambda s, a, t, r: self._pof_rr(s, a),
            EventType.RRF: lambda s, a, t, r: self._pof_rrf(s, a),
            EventType.PRF: lambda s, a, t, r: jnp.array(0.0, dtype=jnp.float32),
            EventType.SC: lambda s, a, t, r: self._pof_sc(s, a),
            EventType.CE: lambda s, a, t, r: self._pof_ce(s, a),
        }

    def calculate_payoff(
        self,
        event_type: Any,
        state: ContractState,
        attributes: ContractAttributes,
        time: ActusDateTime,
        risk_factor_observer: RiskFactorObserver,
        child_contract_observer: Any | None = None,
    ) -> jnp.ndarray:
        """Calculate payoff for NAM event via dict dispatch.

        Args:
            event_type: Type of event
            state: Contract state before event
            attributes: Contract attributes
            time: Event time
            risk_factor_observer: Risk factor observer

        Returns:
            JAX array containing the payoff amount
        """
        handler = self._build_dispatch_table().get(event_type)
        if handler is not None:
            return handler(state, attributes, time, risk_factor_observer)  # type: ignore[no-any-return]
        return jnp.array(0.0, dtype=jnp.float32)

    def _pof_ad(self, state: ContractState, attrs: ContractAttributes) -> jnp.ndarray:
        """POF_AD: Analysis Date - no cashflow."""
        return jnp.array(0.0, dtype=jnp.float32)

    def _pof_ied(self, state: ContractState, attrs: ContractAttributes) -> jnp.ndarray:
        """POF_IED: Initial Exchange - disburse principal.

        Formula: R(CNTRL) × (-1) × Nsc × NT
        """
        role_sign = contract_role_sign(self.contract_role)
        return (
            jnp.array(role_sign * (-1.0), dtype=jnp.float32)
            * state.nsc
            * (attrs.notional_principal or 0.0)
        )

    def _pof_pr(
        self, state: ContractState, attrs: ContractAttributes, time: ActusDateTime
    ) -> jnp.ndarray:
        """POF_PR: Principal Redemption - pay principal NET of accrued interest.

        Formula: Nsc × (Prnxt - Ipac - Y(Sd, t) × Ipnr × Ipcb)
        No R(CNTRL) — all signed state variables.
        """
        yf = year_fraction(state.sd, time, attrs.day_count_convention or DayCountConvention.A360)

        ipcb = state.ipcb if state.ipcb is not None else state.nt
        accrued_interest = yf * state.ipnr * ipcb

        prnxt = state.prnxt or jnp.array(0.0, dtype=jnp.float32)
        net_payment = prnxt - state.ipac - accrued_interest

        # Clamp: don't pay more principal than remaining notional
        net_payment = jnp.where(
            jnp.abs(net_payment) > jnp.abs(state.nt),
            state.nt,
            net_payment,
        )

        return state.nsc * net_payment

    def _pof_md(self, state: ContractState, attrs: ContractAttributes) -> jnp.ndarray:
        """POF_MD: Maturity - final principal + accrued interest + fees.

        Formula: Nsc × Nt + Isc × Ipac + Feac
        No R(CNTRL) — all signed state variables.
        """
        return state.nsc * state.nt + state.isc * state.ipac + state.feac

    def _pof_pp(
        self,
        state: ContractState,
        attrs: ContractAttributes,
        time: ActusDateTime,
        rf_obs: RiskFactorObserver,
    ) -> jnp.ndarray:
        """POF_PP_NAM: Principal Prepayment.

        Formula:
            POF_PP_NAM = X^CURS_CUR(t) × f(O_ev(CID, PP, t))

        The prepayment amount is observed from the risk factor observer.
        """
        try:
            pp_amount = rf_obs.observe_event(
                attrs.contract_id or "",
                EventType.PP,
                time,
                state,
                attrs,
            )
            return jnp.array(float(pp_amount), dtype=jnp.float32)
        except (KeyError, NotImplementedError, TypeError):
            return jnp.array(0.0, dtype=jnp.float32)

    def _pof_py(
        self,
        state: ContractState,
        attrs: ContractAttributes,
        time: ActusDateTime,
        rf_obs: RiskFactorObserver,
    ) -> jnp.ndarray:
        """POF_PY: Penalty payment - observed penalty amount."""
        return jnp.array(0.0, dtype=jnp.float32)

    def _pof_fp(
        self,
        state: ContractState,
        attrs: ContractAttributes,
        time: ActusDateTime,
        rf_obs: RiskFactorObserver,
    ) -> jnp.ndarray:
        """POF_FP: Fee payment - accrued fees.

        Formula: Feac (signed state variable)
        """
        return state.feac

    def _pof_prd(
        self, state: ContractState, attrs: ContractAttributes, time: ActusDateTime
    ) -> jnp.ndarray:
        """POF_PRD: Purchase - purchase price + accrued interest on IPCB.

        Formula: R(CNTRL) × (-1) × (PPRD + Ipac + Y(Sd, t) × Ipnr × Ipcb)
        R(CNTRL) needed because PPRD is an unsigned attribute.
        """
        yf = year_fraction(state.sd, time, attrs.day_count_convention or DayCountConvention.A360)

        ipcb = state.ipcb if state.ipcb is not None else state.nt
        accrued_interest = yf * state.ipnr * ipcb

        role_sign = contract_role_sign(self.contract_role)
        pprd = attrs.price_at_purchase_date or 0.0
        return jnp.array(role_sign * -1.0, dtype=jnp.float32) * (
            jnp.array(pprd, dtype=jnp.float32) + state.ipac + accrued_interest
        )

    def _pof_td(
        self, state: ContractState, attrs: ContractAttributes, time: ActusDateTime
    ) -> jnp.ndarray:
        """POF_TD: Termination - termination price + accrued interest on IPCB.

        Formula: R(CNTRL) × (PTD + Ipac + Y(Sd, t) × Ipnr × Ipcb)
        R(CNTRL) needed because PTD is an unsigned attribute.
        """
        yf = year_fraction(state.sd, time, attrs.day_count_convention or DayCountConvention.A360)

        ipcb = state.ipcb if state.ipcb is not None else state.nt
        accrued_interest = yf * state.ipnr * ipcb

        role_sign = contract_role_sign(self.contract_role)
        ptd = attrs.price_at_termination_date or 0.0
        return jnp.array(role_sign, dtype=jnp.float32) * (
            jnp.array(ptd, dtype=jnp.float32) + state.ipac + accrued_interest
        )

    def _pof_ip(
        self, state: ContractState, attrs: ContractAttributes, time: ActusDateTime
    ) -> jnp.ndarray:
        """POF_IP: Interest Payment - accrued interest on IPCB.

        Formula: Isc × (Ipac + Y(Sd, t) × Ipnr × Ipcb)
        No R(CNTRL) — all signed state variables.
        """
        yf = year_fraction(state.sd, time, attrs.day_count_convention or DayCountConvention.A360)

        ipcb = state.ipcb if state.ipcb is not None else state.nt
        accrued_interest = yf * state.ipnr * ipcb

        return state.isc * (state.ipac + accrued_interest)

    def _pof_ipci(self, state: ContractState, attrs: ContractAttributes) -> jnp.ndarray:
        """POF_IPCI: Interest Capitalization - no cashflow."""
        return jnp.array(0.0, dtype=jnp.float32)

    def _pof_ipcb(self, state: ContractState, attrs: ContractAttributes) -> jnp.ndarray:
        """POF_IPCB: Interest Calculation Base fixing - no cashflow."""
        return jnp.array(0.0, dtype=jnp.float32)

    def _pof_rr(self, state: ContractState, attrs: ContractAttributes) -> jnp.ndarray:
        """POF_RR: Rate Reset - no cashflow."""
        return jnp.array(0.0, dtype=jnp.float32)

    def _pof_rrf(self, state: ContractState, attrs: ContractAttributes) -> jnp.ndarray:
        """POF_RRF: Rate Reset Fixing - no cashflow."""
        return jnp.array(0.0, dtype=jnp.float32)

    def _pof_sc(self, state: ContractState, attrs: ContractAttributes) -> jnp.ndarray:
        """POF_SC: Scaling - no cashflow."""
        return jnp.array(0.0, dtype=jnp.float32)

    def _pof_ce(self, state: ContractState, attrs: ContractAttributes) -> jnp.ndarray:
        """POF_CE: Credit Event - no cashflow."""
        return jnp.array(0.0, dtype=jnp.float32)


class NAMStateTransitionFunction(BaseStateTransitionFunction):
    """State transition function for NAM contracts.

    Implements all NAM state transitions according to ACTUS specification.
    The key difference from LAM is the PR event, which adjusts notional by
    the NET payment amount (payment - interest), allowing notional to increase.

    ACTUS Reference:
        ACTUS v1.1 Section 7.4 - NAM State Transition Functions
    """

    def _build_dispatch_table(self) -> dict[EventType, Any]:
        """Build event type → handler dispatch table."""
        return {
            EventType.AD: self._stf_ad,
            EventType.IED: self._stf_ied,
            EventType.PR: self._stf_pr,
            EventType.MD: self._stf_md,
            EventType.PP: self._stf_pp,
            EventType.PY: self._stf_py,
            EventType.FP: self._stf_fp,
            EventType.PRD: self._stf_prd,
            EventType.TD: self._stf_td,
            EventType.IP: self._stf_ip,
            EventType.IPCI: self._stf_ipci,
            EventType.IPCB: self._stf_ipcb,
            EventType.RR: self._stf_rr,
            EventType.RRF: self._stf_rrf,
            EventType.SC: self._stf_sc,
            EventType.CE: self._stf_ce,
        }

    def transition_state(
        self,
        event_type: Any,
        state: ContractState,
        attributes: ContractAttributes,
        time: ActusDateTime,
        risk_factor_observer: RiskFactorObserver,
        child_contract_observer: Any | None = None,
    ) -> ContractState:
        """Apply state transition for NAM event via dict dispatch.

        Args:
            event_type: Type of event
            state: Contract state before event
            attributes: Contract attributes
            time: Event time
            risk_factor_observer: Observer for market data

        Returns:
            New contract state after event
        """
        handler = self._build_dispatch_table().get(event_type)
        if handler is not None:
            return handler(state, attributes, time, risk_factor_observer)  # type: ignore[no-any-return]
        return state

    def _stf_ad(
        self,
        state: ContractState,
        attrs: ContractAttributes,
        time: ActusDateTime,
        risk_factor_observer: RiskFactorObserver,
        child_contract_observer: Any | None = None,
    ) -> ContractState:
        """STF_AD: Analysis Date - update status date only."""
        return state.replace(sd=time)

    def _stf_ied(
        self,
        state: ContractState,
        attrs: ContractAttributes,
        time: ActusDateTime,
        risk_factor_observer: RiskFactorObserver,
        child_contract_observer: Any | None = None,
    ) -> ContractState:
        """STF_IED: Initial Exchange - initialize all state variables.

        Same as LAM: Initialize Ipcb based on IPCB mode.
        """
        role_sign = contract_role_sign(attrs.contract_role)

        # Determine IPCB (Interest Calculation Base)
        ipcb_mode = attrs.interest_calculation_base or "NT"
        if ipcb_mode == "NTIED":
            # Fixed at IED notional
            ipcb = role_sign * jnp.array(attrs.notional_principal, dtype=jnp.float32)
        elif ipcb_mode == "NT":
            # Track current notional (will be updated at PR events)
            ipcb = role_sign * jnp.array(attrs.notional_principal, dtype=jnp.float32)
        else:  # NTL
            # Will be set at first IPCB event
            ipcb = role_sign * jnp.array(attrs.notional_principal, dtype=jnp.float32)

        # Initialize prnxt (signed with role_sign)
        if attrs.next_principal_redemption_amount is not None:
            prnxt = role_sign * jnp.array(attrs.next_principal_redemption_amount, dtype=jnp.float32)
        else:
            prnxt = state.prnxt or jnp.array(
                0.0, dtype=jnp.float32
            )  # Keep auto-calculated value from initialize_state

        # Use accrued_interest from attributes if provided (signed with role_sign)
        ipac_val = role_sign * attrs.accrued_interest if attrs.accrued_interest is not None else 0.0

        return state.replace(
            sd=time,
            nt=role_sign * jnp.array(attrs.notional_principal or 0.0, dtype=jnp.float32),
            ipnr=jnp.array(attrs.nominal_interest_rate or 0.0, dtype=jnp.float32),
            ipac=jnp.array(ipac_val, dtype=jnp.float32),
            feac=jnp.array(0.0, dtype=jnp.float32),
            nsc=jnp.array(1.0, dtype=jnp.float32),
            isc=jnp.array(1.0, dtype=jnp.float32),
            ipcb=ipcb,
            prnxt=prnxt,
        )

    def _stf_pr(
        self,
        state: ContractState,
        attrs: ContractAttributes,
        time: ActusDateTime,
        risk_factor_observer: RiskFactorObserver,
        child_contract_observer: Any | None = None,
    ) -> ContractState:
        """STF_PR: Principal Redemption - adjust notional by NET payment.

        Formula:
            Nt = Nt - R(CNTRL) × (Prnxt - Ipac - Y(Sd, t) × Ipnr × Ipcb)
            Ipac = 0 (interest paid/capitalized)
            Ipcb = Nt (if IPCB='NT')

        Key Feature: If Prnxt < interest, notional INCREASES (negative amortization).
        """
        yf = year_fraction(state.sd, time, attrs.day_count_convention or DayCountConvention.A360)

        # Calculate accrued interest using current IPCB
        ipcb = state.ipcb if state.ipcb is not None else state.nt
        accrued_interest = yf * state.ipnr * ipcb

        # Total accrued interest (stored in ipac for IP event to pay out)
        new_ipac = state.ipac + accrued_interest

        # Net principal reduction = payment - all interest
        # If negative, this represents principal increase (negative amortization)
        # Per ACTUS spec: Nt_t = Nt_(t-) - (Prnxt - Ipac - Y*Ipnr*Ipcb)
        prnxt = state.prnxt or jnp.array(0.0, dtype=jnp.float32)
        net_principal_reduction = prnxt - new_ipac
        # Clamp: don't reduce notional past zero
        net_principal_reduction = jnp.where(
            jnp.abs(net_principal_reduction) > jnp.abs(state.nt),
            state.nt,
            net_principal_reduction,
        )
        new_nt = state.nt - net_principal_reduction

        # Update IPCB if mode is 'NT'
        ipcb_mode = attrs.interest_calculation_base or "NT"
        new_ipcb: jnp.ndarray
        if ipcb_mode == "NT":
            new_ipcb = new_nt
        elif ipcb_mode == "NTIED":
            new_ipcb = state.ipcb or jnp.array(0.0, dtype=jnp.float32)  # Fixed at IED
        else:  # NTL
            new_ipcb = state.ipcb or jnp.array(
                0.0, dtype=jnp.float32
            )  # Only updated at IPCB events

        return state.replace(
            sd=time,
            nt=new_nt,
            ipac=new_ipac,
            ipcb=new_ipcb,
        )

    def _stf_md(
        self,
        state: ContractState,
        attrs: ContractAttributes,
        time: ActusDateTime,
        risk_factor_observer: RiskFactorObserver,
        child_contract_observer: Any | None = None,
    ) -> ContractState:
        """STF_MD: Maturity - zero out all state variables."""
        return state.replace(
            sd=time,
            nt=jnp.array(0.0, dtype=jnp.float32),
            ipac=jnp.array(0.0, dtype=jnp.float32),
            feac=jnp.array(0.0, dtype=jnp.float32),
            ipcb=jnp.array(0.0, dtype=jnp.float32),
        )

    def _stf_pp(
        self,
        state: ContractState,
        attrs: ContractAttributes,
        time: ActusDateTime,
        risk_factor_observer: RiskFactorObserver,
        child_contract_observer: Any | None = None,
    ) -> ContractState:
        """STF_PP_NAM: Prepayment - accrue interest, reduce notional, update IPCB.

        Updates:
            ipac_t = Ipac_t⁻ + Y(Sd_t⁻, t) × Ipnr_t⁻ × Ipcb_t⁻
            nt_t = Nt_t⁻ - PP_amount
            ipcb_t = Nt_t (if IPCB='NT')
            sd_t = t
        """
        dcc = attrs.day_count_convention or DayCountConvention.A360
        yf = year_fraction(state.sd, time, dcc)

        ipcb = state.ipcb if state.ipcb is not None else state.nt
        new_ipac = state.ipac + yf * state.ipnr * ipcb

        # Get prepayment amount from risk factor observer
        try:
            pp_amount = float(
                risk_factor_observer.observe_event(
                    attrs.contract_id or "",
                    EventType.PP,
                    time,
                    state,
                    attrs,
                )
            )
        except (KeyError, NotImplementedError, TypeError):
            pp_amount = 0.0

        new_nt = state.nt - jnp.array(pp_amount, dtype=jnp.float32)

        # Update IPCB based on mode
        ipcb_mode = attrs.interest_calculation_base or "NT"
        if ipcb_mode in ("NT", "NTIED"):
            new_ipcb = new_nt
        else:  # NTL - only updated at IPCB events
            new_ipcb = state.ipcb or jnp.array(0.0, dtype=jnp.float32)

        return state.replace(
            sd=time,
            nt=new_nt,
            ipac=new_ipac,
            ipcb=new_ipcb,
        )

    def _stf_py(
        self,
        state: ContractState,
        attrs: ContractAttributes,
        time: ActusDateTime,
        risk_factor_observer: RiskFactorObserver,
        child_contract_observer: Any | None = None,
    ) -> ContractState:
        """STF_PY: Penalty - accrue interest and fees."""
        yf = year_fraction(state.sd, time, attrs.day_count_convention or DayCountConvention.A360)
        ipcb = state.ipcb if state.ipcb is not None else state.nt
        new_ipac = state.ipac + yf * state.ipnr * ipcb

        return state.replace(sd=time, ipac=new_ipac)

    def _stf_fp(
        self,
        state: ContractState,
        attrs: ContractAttributes,
        time: ActusDateTime,
        risk_factor_observer: RiskFactorObserver,
        child_contract_observer: Any | None = None,
    ) -> ContractState:
        """STF_FP: Fee Payment - reset accrued fees."""
        yf = year_fraction(state.sd, time, attrs.day_count_convention or DayCountConvention.A360)
        ipcb = state.ipcb if state.ipcb is not None else state.nt
        new_ipac = state.ipac + yf * state.ipnr * ipcb

        return state.replace(sd=time, ipac=new_ipac, feac=jnp.array(0.0, dtype=jnp.float32))

    def _stf_prd(
        self,
        state: ContractState,
        attrs: ContractAttributes,
        time: ActusDateTime,
        risk_factor_observer: RiskFactorObserver,
        child_contract_observer: Any | None = None,
    ) -> ContractState:
        """STF_PRD: Purchase - accrue interest and update status date."""
        yf = year_fraction(state.sd, time, attrs.day_count_convention or DayCountConvention.A360)
        ipcb = state.ipcb if state.ipcb is not None else state.nt
        new_ipac = state.ipac + yf * state.ipnr * ipcb
        return state.replace(sd=time, ipac=new_ipac)

    def _stf_td(
        self,
        state: ContractState,
        attrs: ContractAttributes,
        time: ActusDateTime,
        risk_factor_observer: RiskFactorObserver,
        child_contract_observer: Any | None = None,
    ) -> ContractState:
        """STF_TD: Termination - zero out all states."""
        return state.replace(
            sd=time,
            nt=jnp.array(0.0, dtype=jnp.float32),
            ipac=jnp.array(0.0, dtype=jnp.float32),
            feac=jnp.array(0.0, dtype=jnp.float32),
            ipcb=jnp.array(0.0, dtype=jnp.float32),
        )

    def _stf_ip(
        self,
        state: ContractState,
        attrs: ContractAttributes,
        time: ActusDateTime,
        risk_factor_observer: RiskFactorObserver,
        child_contract_observer: Any | None = None,
    ) -> ContractState:
        """STF_IP: Interest Payment - reset accrued interest."""
        return state.replace(sd=time, ipac=jnp.array(0.0, dtype=jnp.float32))

    def _stf_ipci(
        self,
        state: ContractState,
        attrs: ContractAttributes,
        time: ActusDateTime,
        risk_factor_observer: RiskFactorObserver,
        child_contract_observer: Any | None = None,
    ) -> ContractState:
        """STF_IPCI: Interest Capitalization - add interest to notional.

        Formula:
            Nt = Nt + Ipac + Y(Sd, t) × Ipnr × Ipcb
            Ipac = 0
            Ipcb = Nt (if IPCB='NT')
        """
        yf = year_fraction(state.sd, time, attrs.day_count_convention or DayCountConvention.A360)
        ipcb = state.ipcb if state.ipcb is not None else state.nt
        accrued = yf * state.ipnr * ipcb

        new_nt = state.nt + state.ipac + accrued

        # Update IPCB if mode is 'NT'
        ipcb_mode = attrs.interest_calculation_base or "NT"
        new_ipcb = (
            new_nt if ipcb_mode == "NT" else (state.ipcb or jnp.array(0.0, dtype=jnp.float32))
        )

        return state.replace(
            sd=time, nt=new_nt, ipac=jnp.array(0.0, dtype=jnp.float32), ipcb=new_ipcb
        )

    def _stf_ipcb(
        self,
        state: ContractState,
        attrs: ContractAttributes,
        time: ActusDateTime,
        risk_factor_observer: RiskFactorObserver,
        child_contract_observer: Any | None = None,
    ) -> ContractState:
        """STF_IPCB: Interest Calculation Base fixing - reset IPCB to current notional.

        Formula:
            Ipcb = Nt
            Ipac = Ipac + Y(Sd, t) × Ipnr × Ipcb_old
        """
        yf = year_fraction(state.sd, time, attrs.day_count_convention or DayCountConvention.A360)
        ipcb_old = state.ipcb if state.ipcb is not None else state.nt
        new_ipac = state.ipac + yf * state.ipnr * ipcb_old

        return state.replace(sd=time, ipcb=state.nt, ipac=new_ipac)

    def _stf_rr(
        self,
        state: ContractState,
        attrs: ContractAttributes,
        time: ActusDateTime,
        risk_factor_observer: RiskFactorObserver,
        child_contract_observer: Any | None = None,
    ) -> ContractState:
        """STF_RR: Rate Reset - observe market rate and apply caps/floors.

        Formula:
            Ipac = Ipac + Y(Sd, t) * Ipnr * Ipcb
            Ipnr = min(max(RRMLT * O_rf(RRMO, t) + RRSP, RRLF), RRLC)
        """
        yf = year_fraction(state.sd, time, attrs.day_count_convention or DayCountConvention.A360)
        ipcb = state.ipcb if state.ipcb is not None else state.nt
        new_ipac = state.ipac + yf * state.ipnr * ipcb

        # Observe market rate
        market_object = attrs.rate_reset_market_object or ""
        observed_rate = float(
            risk_factor_observer.observe_risk_factor(market_object, time, state, attrs)
        )

        # Apply multiplier and spread
        multiplier = attrs.rate_reset_multiplier if attrs.rate_reset_multiplier is not None else 1.0
        spread = attrs.rate_reset_spread if attrs.rate_reset_spread is not None else 0.0
        new_rate = multiplier * observed_rate + spread

        # Apply floor and cap
        if attrs.rate_reset_floor is not None:
            new_rate = max(new_rate, attrs.rate_reset_floor)
        if attrs.rate_reset_cap is not None:
            new_rate = min(new_rate, attrs.rate_reset_cap)

        return state.replace(
            sd=time,
            ipac=new_ipac,
            ipnr=jnp.array(new_rate, dtype=jnp.float32),
        )

    def _stf_rrf(
        self,
        state: ContractState,
        attrs: ContractAttributes,
        time: ActusDateTime,
        risk_factor_observer: RiskFactorObserver,
        child_contract_observer: Any | None = None,
    ) -> ContractState:
        """STF_RRF: Rate Reset Fixing - fix interest rate to next reset rate.

        Formula:
            Ipac = Ipac + Y(Sd, t) * Ipnr * Ipcb
            Ipnr = RRNXT
        """
        yf = year_fraction(state.sd, time, attrs.day_count_convention or DayCountConvention.A360)
        ipcb = state.ipcb if state.ipcb is not None else state.nt
        new_ipac = state.ipac + yf * state.ipnr * ipcb

        new_rate = attrs.rate_reset_next if attrs.rate_reset_next is not None else state.ipnr

        return state.replace(
            sd=time,
            ipac=new_ipac,
            ipnr=jnp.array(float(new_rate), dtype=jnp.float32),
        )

    def _stf_sc(
        self,
        state: ContractState,
        attrs: ContractAttributes,
        time: ActusDateTime,
        risk_factor_observer: RiskFactorObserver,
        child_contract_observer: Any | None = None,
    ) -> ContractState:
        """STF_SC: Scaling - update scaling multipliers."""
        yf = year_fraction(state.sd, time, attrs.day_count_convention or DayCountConvention.A360)
        ipcb = state.ipcb if state.ipcb is not None else state.nt
        new_ipac = state.ipac + yf * state.ipnr * ipcb

        return state.replace(sd=time, ipac=new_ipac)

    def _stf_ce(
        self,
        state: ContractState,
        attrs: ContractAttributes,
        time: ActusDateTime,
        risk_factor_observer: RiskFactorObserver,
        child_contract_observer: Any | None = None,
    ) -> ContractState:
        """STF_CE: Credit Event - update status date."""
        return state.replace(sd=time)


class NegativeAmortizerContract(BaseContract):
    """Negative Amortizer (NAM) contract.

    Amortizing loan where principal can increase when payments are less than
    accrued interest (negative amortization). Extends LAM pattern with modified
    principal redemption handling.

    ACTUS Reference:
        ACTUS v1.1 Section 7.4 - NAM: Negative Amortizer

    Attributes:
        attributes: Contract attributes
        risk_factor_observer: Risk factor observer for market data

    Example:
        See module docstring for usage example.
    """

    def __init__(
        self,
        attributes: ContractAttributes,
        risk_factor_observer: RiskFactorObserver,
        child_contract_observer: Any | None = None,
    ):
        """Initialize NAM contract.

        Args:
            attributes: Contract attributes
            risk_factor_observer: Risk factor observer

        Raises:
            ValueError: If contract_type is not NAM
            ValueError: If required attributes missing
        """
        if attributes.contract_type != ContractType.NAM:
            raise ValueError(f"Contract type must be NAM, got {attributes.contract_type}")

        # Validate required attributes
        if not attributes.initial_exchange_date:
            raise ValueError("initial_exchange_date required for NAM")
        if not attributes.principal_redemption_cycle and not attributes.maturity_date:
            raise ValueError("Either principal_redemption_cycle or maturity_date required")

        super().__init__(
            attributes=attributes,
            risk_factor_observer=risk_factor_observer,
            child_contract_observer=child_contract_observer,
        )

    def generate_event_schedule(self) -> EventSchedule:
        """Generate NAM event schedule per ACTUS specification.

        Same as LAM schedule, with all event types supported.

        Events before status_date are excluded.
        """
        events: list[ContractEvent] = []
        attrs = self.attributes
        ied = attrs.initial_exchange_date
        md = attrs.maturity_date
        sd = attrs.status_date
        currency = attrs.currency or "XXX"
        assert ied is not None

        bdc = attrs.business_day_convention
        eomc = attrs.end_of_month_convention
        cal = attrs.calendar

        def _sched(
            anchor: ActusDateTime, cycle: str, end: ActusDateTime | None
        ) -> list[ActusDateTime]:
            return generate_schedule(
                start=anchor,
                cycle=cycle,
                end=end,
                end_of_month_convention=eomc or EndOfMonthConvention.SD,
                business_day_convention=bdc or BusinessDayConvention.NULL,
                calendar=cal or Calendar.NO_CALENDAR,
            )

        def _add(etype: EventType, time: ActusDateTime) -> None:
            events.append(
                ContractEvent(
                    event_type=etype,
                    event_time=time,
                    payoff=jnp.array(0.0, dtype=jnp.float32),
                    currency=currency,
                    sequence=0,
                )
            )

        # IED: only if IED >= SD
        if ied >= sd:
            _add(EventType.IED, ied)

        # PR: Principal Redemption schedule
        if attrs.principal_redemption_cycle:
            pr_anchor = attrs.principal_redemption_anchor or ied
            pr_dates = _sched(pr_anchor, attrs.principal_redemption_cycle, md)
            # Long stub: remove last PR date before MD if it's not on cycle end
            pr_cycle_str = attrs.principal_redemption_cycle or ""
            if pr_cycle_str.endswith("+") and pr_dates and md and pr_dates[-1] != md:
                pr_dates = pr_dates[:-1]
            for dt in pr_dates:
                if md and dt >= md:
                    break
                if dt >= ied:
                    _add(EventType.PR, dt)

        # IP: Interest Payment schedule from IPANX (or IED)
        if attrs.interest_payment_cycle:
            ip_anchor = attrs.interest_payment_anchor or ied
            ipced = attrs.interest_capitalization_end_date
            ip_dates = _sched(ip_anchor, attrs.interest_payment_cycle, md)
            if ipced and ipced not in ip_dates:
                ip_dates = sorted(set(ip_dates + [ipced]))
            ip_cycle_str = attrs.interest_payment_cycle or ""
            if md and md not in ip_dates and ip_dates:
                if ip_cycle_str.endswith("+"):
                    ip_dates[-1] = md
                else:
                    ip_dates.append(md)
                ip_dates = sorted(set(ip_dates))
            for dt in ip_dates:
                if dt < ied:
                    continue
                if ipced and dt <= ipced:
                    _add(EventType.IPCI, dt)
                else:
                    _add(EventType.IP, dt)

        # IPCB: Interest Calculation Base schedule (only if IPCB='NTL')
        if attrs.interest_calculation_base == "NTL" and attrs.interest_calculation_base_cycle:
            ipcb_anchor = attrs.interest_calculation_base_anchor or ied
            ipcb_dates = _sched(ipcb_anchor, attrs.interest_calculation_base_cycle, md)
            for dt in ipcb_dates:
                if dt > ied:
                    _add(EventType.IPCB, dt)

        # RR: Rate Reset schedule (exclude events at MD, handle long stub)
        if attrs.rate_reset_cycle and attrs.rate_reset_anchor:
            rr_dates = _sched(attrs.rate_reset_anchor, attrs.rate_reset_cycle, md)
            rr_cycle_str = attrs.rate_reset_cycle or ""
            if rr_cycle_str.endswith("+") and rr_dates and md and rr_dates[-1] != md:
                rr_dates = rr_dates[:-1]
            first_rr = True
            for dt in rr_dates:
                if md and dt >= md:
                    break
                if first_rr and attrs.rate_reset_next is not None:
                    _add(EventType.RRF, dt)
                    first_rr = False
                else:
                    _add(EventType.RR, dt)
                    first_rr = False

        # FP: Fee Payment schedule
        if attrs.fee_payment_cycle:
            fp_anchor = attrs.fee_payment_anchor or ied
            fp_dates = _sched(fp_anchor, attrs.fee_payment_cycle, md)
            for dt in fp_dates:
                if dt > ied:
                    _add(EventType.FP, dt)

        # SC: Scaling Index schedule
        if attrs.scaling_index_cycle:
            sc_anchor = attrs.scaling_index_anchor or ied
            sc_dates = _sched(sc_anchor, attrs.scaling_index_cycle, md)
            for dt in sc_dates:
                if dt > ied:
                    _add(EventType.SC, dt)

        # PRD: Purchase date
        if attrs.purchase_date:
            _add(EventType.PRD, attrs.purchase_date)

        # TD: Termination date
        if attrs.termination_date:
            _add(EventType.TD, attrs.termination_date)

        # MD: Maturity Date
        if md:
            _add(EventType.MD, md)

        # Filter out events before SD, sort
        events = [e for e in events if e.event_time >= sd]

        # If PRD exists, remove IED and non-PRD events before/at PRD
        if attrs.purchase_date:
            prd_time = attrs.purchase_date
            events = [
                e
                for e in events
                if e.event_type == EventType.PRD
                or (e.event_type != EventType.IED and e.event_time > prd_time)
            ]

        events.sort(
            key=lambda e: (e.event_time.to_iso(), EVENT_SCHEDULE_PRIORITY.get(e.event_type, 99))
        )

        # If TD exists, remove all events after TD
        if attrs.termination_date:
            td_time = attrs.termination_date
            events = [e for e in events if e.event_time <= td_time]

        # Reassign sequence numbers
        for i in range(len(events)):
            events[i] = ContractEvent(
                event_type=events[i].event_type,
                event_time=events[i].event_time,
                payoff=events[i].payoff,
                currency=events[i].currency,
                sequence=i,
            )

        return EventSchedule(events=tuple(events), contract_id=attrs.contract_id)

    def _pre_simulate_to_prd(self, attrs: ContractAttributes, prnxt: jnp.ndarray) -> ContractState:
        """Pre-simulate events from IED to PRD to get correct initial state."""
        ied = attrs.initial_exchange_date
        prd = attrs.purchase_date
        sd = attrs.status_date
        md = attrs.maturity_date
        assert ied is not None
        assert prd is not None

        bdc = attrs.business_day_convention
        eomc = attrs.end_of_month_convention
        cal = attrs.calendar

        def _sched(
            anchor: ActusDateTime | None, cycle: str | None, end: ActusDateTime | None
        ) -> list[ActusDateTime]:
            if anchor is None or cycle is None or end is None:
                return []
            return generate_schedule(
                start=anchor,
                cycle=cycle,
                end=end,
                end_of_month_convention=eomc or EndOfMonthConvention.SD,
                business_day_convention=bdc or BusinessDayConvention.NULL,
                calendar=cal or Calendar.NO_CALENDAR,
            )

        pre_events: list[tuple[ActusDateTime, EventType]] = []
        pre_events.append((ied, EventType.IED))

        if attrs.principal_redemption_cycle:
            pr_anchor = attrs.principal_redemption_anchor or ied
            pr_dates = _sched(pr_anchor, attrs.principal_redemption_cycle, md or prd)
            for dt in pr_dates:
                if dt >= ied and dt <= prd:
                    pre_events.append((dt, EventType.PR))

        if attrs.interest_payment_cycle:
            ip_anchor = attrs.interest_payment_anchor or ied
            ipced = attrs.interest_capitalization_end_date
            ip_dates = _sched(ip_anchor, attrs.interest_payment_cycle, md or prd)
            for dt in ip_dates:
                if dt < ied or dt > prd:
                    continue
                if ipced and dt <= ipced:
                    pre_events.append((dt, EventType.IPCI))
                else:
                    pre_events.append((dt, EventType.IP))

        if attrs.rate_reset_cycle and attrs.rate_reset_anchor:
            rr_dates = _sched(attrs.rate_reset_anchor, attrs.rate_reset_cycle, md or prd)
            first_rr = True
            for dt in rr_dates:
                if dt > prd:
                    break
                if first_rr and attrs.rate_reset_next is not None:
                    pre_events.append((dt, EventType.RRF))
                    first_rr = False
                else:
                    pre_events.append((dt, EventType.RR))
                    first_rr = False

        if attrs.interest_calculation_base == "NTL" and attrs.interest_calculation_base_cycle:
            ipcb_anchor = attrs.interest_calculation_base_anchor or ied
            ipcb_dates = _sched(ipcb_anchor, attrs.interest_calculation_base_cycle, md or prd)
            for dt in ipcb_dates:
                if dt > ied and dt <= prd:
                    pre_events.append((dt, EventType.IPCB))

        pre_events = [(t, e) for t, e in pre_events if t >= ied]
        pre_events.sort(key=lambda e: (e[0].to_iso(), EVENT_SCHEDULE_PRIORITY.get(e[1], 99)))

        state = ContractState(
            sd=ied,
            tmd=md or ied,
            nt=jnp.array(0.0, dtype=jnp.float32),
            ipnr=jnp.array(0.0, dtype=jnp.float32),
            ipac=jnp.array(0.0, dtype=jnp.float32),
            feac=jnp.array(0.0, dtype=jnp.float32),
            nsc=jnp.array(1.0, dtype=jnp.float32),
            isc=jnp.array(1.0, dtype=jnp.float32),
            prnxt=prnxt,
            ipcb=jnp.array(0.0, dtype=jnp.float32),
        )

        stf = self.get_state_transition_function(None)
        for time, etype in pre_events:
            state = stf.transition_state(
                event_type=etype,
                state=state,
                attributes=attrs,
                time=time,
                risk_factor_observer=self.risk_factor_observer,
            )

        if ied < sd and sd < prd:
            state = state.replace(sd=sd, ipac=jnp.array(0.0, dtype=jnp.float32))

        return state

    def initialize_state(self) -> ContractState:
        """Initialize NAM contract state.

        When IED < SD (contract already existed), state is initialized
        as if STF_IED already ran.

        Returns:
            Initial contract state
        """
        attrs = self.attributes
        sd = attrs.status_date
        ied = attrs.initial_exchange_date
        role_sign = contract_role_sign(attrs.contract_role)

        # Initialize Prnxt (next principal redemption amount)
        prnxt_val = attrs.next_principal_redemption_amount or 0.0
        prnxt = jnp.array(role_sign * prnxt_val, dtype=jnp.float32)

        # PRD pre-simulation
        if attrs.purchase_date and ied:
            return self._pre_simulate_to_prd(attrs, prnxt)

        needs_post_ied = ied and ied < sd
        if needs_post_ied:
            assert ied is not None
            # Contract already started - initialize post-IED state
            nt = role_sign * (attrs.notional_principal or 0.0)
            ipnr = attrs.nominal_interest_rate or 0.0
            dcc = attrs.day_count_convention or DayCountConvention.A360
            accrual_start = attrs.interest_payment_anchor or ied
            if attrs.accrued_interest is not None:
                ipac = attrs.accrued_interest
            else:
                yf = year_fraction(accrual_start, sd, dcc)
                ipac = yf * ipnr * abs(nt)
            ipcb_val = abs(nt)
            return ContractState(
                sd=sd,
                tmd=attrs.maturity_date or sd,
                nt=jnp.array(nt, dtype=jnp.float32),
                ipnr=jnp.array(ipnr, dtype=jnp.float32),
                ipac=jnp.array(ipac, dtype=jnp.float32),
                feac=jnp.array(0.0, dtype=jnp.float32),
                nsc=jnp.array(1.0, dtype=jnp.float32),
                isc=jnp.array(1.0, dtype=jnp.float32),
                prnxt=prnxt,
                ipcb=jnp.array(ipcb_val, dtype=jnp.float32),
            )

        return ContractState(
            sd=sd,
            tmd=attrs.maturity_date or sd,
            nt=jnp.array(0.0, dtype=jnp.float32),  # Set at IED
            ipnr=jnp.array(0.0, dtype=jnp.float32),  # Set at IED
            ipac=jnp.array(0.0, dtype=jnp.float32),
            feac=jnp.array(0.0, dtype=jnp.float32),
            nsc=jnp.array(1.0, dtype=jnp.float32),
            isc=jnp.array(1.0, dtype=jnp.float32),
            prnxt=prnxt,
            ipcb=jnp.array(0.0, dtype=jnp.float32),  # Set at IED
        )

    def get_payoff_function(self, event_type: Any) -> NAMPayoffFunction:
        """Get NAM payoff function.

        Args:
            event_type: Type of event (not used, all events use same POF)

        Returns:
            NAM payoff function instance
        """
        return NAMPayoffFunction(
            contract_role=self.attributes.contract_role,
            currency=self.attributes.currency,
            settlement_currency=None,
        )

    def get_state_transition_function(self, event_type: Any) -> NAMStateTransitionFunction:
        """Get NAM state transition function.

        Args:
            event_type: Type of event (not used, all events use same STF)

        Returns:
            NAM state transition function instance
        """
        return NAMStateTransitionFunction()
