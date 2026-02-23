"""Linear Amortizer (LAM) contract implementation.

This module implements the LAM contract type - an amortizing loan with fixed
principal redemption amounts where principal is repaid in regular installments.
LAM is the foundation for other amortizing contracts (NAM, ANN, LAX).

ACTUS Reference:
    ACTUS v1.1 Section 7.2 - LAM: Linear Amortizer

Key Features:
    - Fixed principal redemption amounts (Prnxt)
    - Regular principal reduction (PR events)
    - Interest calculated on IPCB (Interest Calculation Base)
    - Three IPCB modes: NT (notional tracking), NTIED (fixed at IED), NTL (lagged)
    - Optional IPCB events for base fixing
    - Maturity can be calculated if not provided
    - 16 event types total

IPCB Modes:
    - NT: Interest calculated on current notional (lagging one period)
    - NTIED: Interest calculated on initial notional at IED (fixed)
    - NTL: Interest calculated on notional at last IPCB event (lagged with updates)

Example:
    >>> from jactus.contracts import create_contract
    >>> from jactus.core import ContractAttributes, ContractType, ContractRole
    >>> from jactus.core import ActusDateTime, DayCountConvention
    >>> from jactus.observers import ConstantRiskFactorObserver
    >>>
    >>> attrs = ContractAttributes(
    ...     contract_id="MORTGAGE-001",
    ...     contract_type=ContractType.LAM,
    ...     contract_role=ContractRole.RPA,
    ...     status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
    ...     initial_exchange_date=ActusDateTime(2024, 1, 15, 0, 0, 0),
    ...     maturity_date=ActusDateTime(2054, 1, 15, 0, 0, 0),  # 30 years
    ...     currency="USD",
    ...     notional_principal=300000.0,
    ...     nominal_interest_rate=0.065,
    ...     day_count_convention=DayCountConvention.A360,
    ...     principal_redemption_cycle="1M",  # Monthly payments
    ...     next_principal_redemption_amount=1000.0,  # $1000/month principal
    ...     interest_calculation_base="NT"  # Interest on current notional
    ... )
    >>>
    >>> rf_obs = ConstantRiskFactorObserver(constant_value=0.065)
    >>> contract = create_contract(attrs, rf_obs)
    >>> result = contract.simulate()
"""

import math
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


class LAMPayoffFunction(BasePayoffFunction):
    """Payoff function for LAM contracts.

    Implements all LAM payoff functions according to ACTUS specification.
    The key difference from PAM is the PR (Principal Redemption) event and
    the use of IPCB (Interest Calculation Base) for interest calculations.

    ACTUS Reference:
        ACTUS v1.1 Section 7.2 - LAM Payoff Functions

    Events:
        AD: Analysis Date (0.0)
        IED: Initial Exchange Date (disburse principal)
        PR: Principal Redemption (fixed principal payment)
        MD: Maturity Date (final principal + interest)
        PP: Principal Prepayment
        PY: Penalty Payment
        FP: Fee Payment
        PRD: Purchase Date
        TD: Termination Date
        IP: Interest Payment (on IPCB, not current notional)
        IPCI: Interest Capitalization
        IPCB: Interest Calculation Base fixing
        RR: Rate Reset
        RRF: Rate Reset Fixing
        SC: Scaling
        CE: Credit Event
    """

    def _build_dispatch_table(self) -> dict[EventType, Any]:
        """Build event type → handler dispatch table.

        Lambdas normalize varying handler signatures to a uniform
        (state, attributes, time, risk_factor_observer) interface.
        """
        return {
            EventType.AD: lambda s, a, t, r: self._pof_ad(s, a),
            EventType.IED: lambda s, a, t, r: self._pof_ied(s, a),
            EventType.PR: lambda s, a, t, r: self._pof_pr(s, a),
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
    ) -> jnp.ndarray:
        """Calculate payoff for LAM event via dict dispatch.

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

    def _pof_pr(self, state: ContractState, attrs: ContractAttributes) -> jnp.ndarray:
        """POF_PR: Principal Redemption - pay fixed principal amount.

        Formula: Nsc × Prnxt (capped at remaining notional)
        No R(CNTRL) — Prnxt is a signed state variable.
        """
        # Cap redemption at remaining notional to avoid overshoot
        prnxt = state.prnxt or jnp.array(0.0, dtype=jnp.float32)
        effective_prnxt = jnp.sign(prnxt) * jnp.minimum(jnp.abs(prnxt), jnp.abs(state.nt))
        return state.nsc * effective_prnxt

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
        """POF_PP_LAM: Principal Prepayment.

        Formula:
            POF_PP_LAM = X^CURS_CUR(t) × f(O_ev(CID, PP, t))

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
        """POF_PY: Penalty payment - observed penalty amount.

        Note: Not yet implemented.
        """
        return jnp.array(0.0, dtype=jnp.float32)

    def _pof_fp(
        self,
        state: ContractState,
        attrs: ContractAttributes,
        time: ActusDateTime,
        rf_obs: RiskFactorObserver,
    ) -> jnp.ndarray:
        """POF_FP: Fee payment - accrued fees.

        Formula: Feac  (signed state variable)
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


class LAMStateTransitionFunction(BaseStateTransitionFunction):
    """State transition function for LAM contracts.

    Implements all LAM state transitions according to ACTUS specification.
    The key differences from PAM are PR event handling and IPCB tracking.

    ACTUS Reference:
        ACTUS v1.1 Section 7.2 - LAM State Transition Functions
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
    ) -> ContractState:
        """Apply state transition for LAM event via dict dispatch.

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
        """STF_IED: Initial Exchange - initialize all state variables.

        Key Feature: Initialize Ipcb based on IPCB mode.
        """
        role_sign = contract_role_sign(attrs.contract_role)

        # Determine IPCB (Interest Calculation Base)
        if attrs.interest_calculation_base_amount is not None:
            # IPCBA overrides: fixed at specified amount
            ipcb = role_sign * jnp.array(attrs.interest_calculation_base_amount, dtype=jnp.float32)
        else:
            # Default: initialize to notional (mode-specific updates happen later)
            ipcb = role_sign * jnp.array(attrs.notional_principal, dtype=jnp.float32)

        # Initialize prnxt (signed with role_sign)
        # If PRNXT is provided, use it. Otherwise, preserve the auto-calculated
        # value from initialize_state (stored in state.prnxt).
        if attrs.next_principal_redemption_amount is not None:
            prnxt = role_sign * jnp.array(attrs.next_principal_redemption_amount, dtype=jnp.float32)
        else:
            prnxt = state.prnxt or jnp.array(0.0, dtype=jnp.float32)  # Keep auto-calculated value

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
    ) -> ContractState:
        """STF_PR: Principal Redemption - reduce notional, update IPCB if needed.

        Formula:
            Nt = Nt - Prnxt  (Prnxt is signed)
            Ipac = Ipac + Y(Sd, t) × Ipnr × Ipcb
            Ipcb = Nt (if IPCB='NT')
        """
        yf = year_fraction(state.sd, time, attrs.day_count_convention or DayCountConvention.A360)

        # Calculate accrued interest using current IPCB
        ipcb = state.ipcb if state.ipcb is not None else state.nt
        new_ipac = state.ipac + yf * state.ipnr * ipcb

        # Reduce notional (prnxt is signed), cap at remaining notional
        prnxt = state.prnxt or jnp.array(0.0, dtype=jnp.float32)
        effective_prnxt = jnp.sign(prnxt) * jnp.minimum(jnp.abs(prnxt), jnp.abs(state.nt))
        new_nt = state.nt - effective_prnxt

        # Update IPCB based on mode
        ipcb_mode = attrs.interest_calculation_base or "NT"
        new_ipcb: jnp.ndarray
        if ipcb_mode in ("NT", "NTIED"):
            new_ipcb = new_nt  # Track current notional
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
    ) -> ContractState:
        """STF_PP_LAM: Prepayment - accrue interest, reduce notional, update IPCB.

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
    ) -> ContractState:
        """STF_PY: Penalty - accrue interest and fees.

        Note: Not yet fully implemented.
        """
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
    ) -> ContractState:
        """STF_FP: Fee Payment - reset accrued fees.

        Note: Not yet fully implemented.
        """
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
    ) -> ContractState:
        """STF_IP: Interest Payment - reset accrued interest."""
        return state.replace(sd=time, ipac=jnp.array(0.0, dtype=jnp.float32))

    def _stf_ipci(
        self,
        state: ContractState,
        attrs: ContractAttributes,
        time: ActusDateTime,
        risk_factor_observer: RiskFactorObserver,
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
        new_ipcb: jnp.ndarray
        if ipcb_mode == "NT":
            new_ipcb = new_nt
        else:
            new_ipcb = state.ipcb or jnp.array(0.0, dtype=jnp.float32)

        return state.replace(
            sd=time, nt=new_nt, ipac=jnp.array(0.0, dtype=jnp.float32), ipcb=new_ipcb
        )

    def _stf_ipcb(
        self,
        state: ContractState,
        attrs: ContractAttributes,
        time: ActusDateTime,
        risk_factor_observer: RiskFactorObserver,
    ) -> ContractState:
        """STF_IPCB: Interest Calculation Base fixing - reset IPCB to current notional.

        Formula:
            Ipcb = Nt
            Ipac = Ipac + Y(Sd, t) × Ipnr × Ipcb_old

        Key Feature: Only used when IPCB='NTL'.
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
    ) -> ContractState:
        """STF_SC: Scaling - update scaling multipliers from index.

        Formula:
            Ipac = Ipac + Y(Sd, t) × Ipnr × Ipcb
            scaling_ratio = I(t) / I_ref
            If SCEF[0] == 'I': Isc = scaling_ratio
            If SCEF[1] == 'N': Nsc = scaling_ratio
        """
        yf = year_fraction(state.sd, time, attrs.day_count_convention or DayCountConvention.A360)
        ipcb = state.ipcb if state.ipcb is not None else state.nt
        new_ipac = state.ipac + yf * state.ipnr * ipcb

        # Observe current scaling index value
        new_isc = state.isc
        new_nsc = state.nsc
        scaling_mo = attrs.scaling_market_object
        if scaling_mo:
            current_index = float(
                risk_factor_observer.observe_risk_factor(scaling_mo, time, state, attrs)
            )
            ref_index = attrs.scaling_index_at_contract_deal_date or 1.0
            if ref_index != 0.0:
                scaling_ratio = current_index / ref_index
            else:
                scaling_ratio = 1.0

            effect_str = str(attrs.scaling_effect.value) if attrs.scaling_effect else "000"
            if len(effect_str) >= 1 and effect_str[0] == "I":
                new_isc = jnp.array(scaling_ratio, dtype=jnp.float32)
            if len(effect_str) >= 2 and effect_str[1] == "N":
                new_nsc = jnp.array(scaling_ratio, dtype=jnp.float32)

        return state.replace(sd=time, ipac=new_ipac, isc=new_isc, nsc=new_nsc)

    def _stf_ce(
        self,
        state: ContractState,
        attrs: ContractAttributes,
        time: ActusDateTime,
        risk_factor_observer: RiskFactorObserver,
    ) -> ContractState:
        """STF_CE: Credit Event - update status date."""
        return state.replace(sd=time)


class LinearAmortizerContract(BaseContract):
    """Linear Amortizer (LAM) contract.

    Amortizing loan with fixed principal redemption amounts. Principal is repaid
    in regular installments (PR events), with interest calculated on an Interest
    Calculation Base (IPCB) that can track current notional, stay fixed, or update
    periodically.

    ACTUS Reference:
        ACTUS v1.1 Section 7.2 - LAM: Linear Amortizer

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
        """Initialize LAM contract.

        Args:
            attributes: Contract attributes
            risk_factor_observer: Risk factor observer
            child_contract_observer: Optional child contract observer

        Raises:
            ValueError: If contract_type is not LAM
            ValueError: If required attributes missing
        """
        if attributes.contract_type != ContractType.LAM:
            raise ValueError(f"Contract type must be LAM, got {attributes.contract_type}")

        # Validate required attributes
        if not attributes.initial_exchange_date:
            raise ValueError("initial_exchange_date required for LAM")
        if not attributes.principal_redemption_cycle and not attributes.maturity_date:
            raise ValueError("Either principal_redemption_cycle or maturity_date required")
        # Note: PRNXT validation is done at simulation time, not contract creation
        # if not attributes.next_principal_redemption_amount:
        #     raise ValueError("next_principal_redemption_amount (PRNXT) required for LAM")

        super().__init__(
            attributes=attributes,
            risk_factor_observer=risk_factor_observer,
            child_contract_observer=child_contract_observer,
        )

    def generate_event_schedule(self) -> EventSchedule:
        """Generate LAM event schedule per ACTUS specification.

        Schedule formula for each event type:
            IED: Single event at initial_exchange_date (if IED >= SD)
            PR:  S(PRANX, PRCL, MD) - principal redemption schedule
            IP:  S(IPANX, IPCL, MD) - interest payment schedule
            IPCI: S(IPANX, IPCL, IPCED) - interest capitalization until end date
            IPCB: S(IPCBANX, IPCBCL, MD) - if IPCB='NTL'
            RR:  S(RRANX, RRCL, MD) - rate reset schedule
            FP:  S(FEANX, FECL, MD) - fee payment schedule
            PRD: Single event at purchase_date
            TD:  Single event at termination_date (truncates schedule)
            MD:  Single event at maturity_date

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

        # Calculate MD if not provided: MD = last PR date from schedule
        if md is None and attrs.principal_redemption_cycle:
            prnxt = attrs.next_principal_redemption_amount or 0.0
            nt = attrs.notional_principal or 0.0
            if prnxt > 0:
                n_periods = math.ceil(nt / prnxt)
                pr_anchor = attrs.principal_redemption_anchor or ied
                # Generate enough dates to find MD
                far_end = pr_anchor.add_period(f"{(n_periods + 2) * 12}M", EndOfMonthConvention.SD)
                pr_dates = generate_schedule(
                    start=pr_anchor,
                    cycle=attrs.principal_redemption_cycle,
                    end=far_end,
                    end_of_month_convention=eomc or EndOfMonthConvention.SD,
                    business_day_convention=bdc or BusinessDayConvention.NULL,
                    calendar=cal or Calendar.NO_CALENDAR,
                )
                pr_dates = [d for d in pr_dates if d >= ied]
                if len(pr_dates) >= n_periods:
                    md = pr_dates[n_periods - 1]

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
            # Stub handling: if MD not on cycle, add MD (and remove last for long stub)
            ip_cycle_str = attrs.interest_payment_cycle or ""
            if md and md not in ip_dates and ip_dates:
                if ip_cycle_str.endswith("+"):
                    # Long stub: replace last cycle date with MD
                    ip_dates[-1] = md
                else:
                    # Short stub (default): keep last cycle date and add MD
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
            # Long stub: remove last date before MD
            ipcb_cycle_str = attrs.interest_calculation_base_cycle or ""
            if ipcb_cycle_str.endswith("+") and ipcb_dates and md and ipcb_dates[-1] != md:
                ipcb_dates = ipcb_dates[:-1]
            for dt in ipcb_dates:
                if dt > ied and (not md or dt < md):
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
        """Pre-simulate events from IED to PRD to get correct initial state.

        When a contract has a purchase date, events between IED and PRD
        (PR, IP, RR, IPCB, etc.) affect the state. This method runs those
        events to compute the true state at purchase time.
        """
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

        # Build pre-PRD event list (IED through events strictly before PRD)
        pre_events: list[tuple[ActusDateTime, EventType]] = []
        pre_events.append((ied, EventType.IED))

        # PR events (include events at PRD time - they reduce NT before purchase)
        if attrs.principal_redemption_cycle:
            pr_anchor = attrs.principal_redemption_anchor or ied
            pr_dates = _sched(pr_anchor, attrs.principal_redemption_cycle, md or prd)
            for dt in pr_dates:
                if dt >= ied and dt <= prd:
                    pre_events.append((dt, EventType.PR))

        # IP events (include events at PRD time - they reset ipac before purchase)
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

        # RR events (include events at PRD time)
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

        # IPCB events
        if attrs.interest_calculation_base == "NTL" and attrs.interest_calculation_base_cycle:
            ipcb_anchor = attrs.interest_calculation_base_anchor or ied
            ipcb_dates = _sched(ipcb_anchor, attrs.interest_calculation_base_cycle, md or prd)
            for dt in ipcb_dates:
                if dt > ied and dt <= prd:
                    pre_events.append((dt, EventType.IPCB))

        # Only keep events from IED onward (some anchors may be before IED)
        pre_events = [(t, e) for t, e in pre_events if t >= ied]

        # Sort by time, then by event priority
        pre_events.sort(key=lambda e: (e[0].to_iso(), EVENT_SCHEDULE_PRIORITY.get(e[1], 99)))

        # Create initial state (pre-IED)
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

        # Run STF for each pre-PRD event
        stf = self.get_state_transition_function(None)
        for time, etype in pre_events:
            state = stf.transition_state(
                event_type=etype,
                state=state,
                attributes=attrs,
                time=time,
                risk_factor_observer=self.risk_factor_observer,
            )

        # When IED < SD, advance state to SD (accrual before SD is not reported)
        # Note: ipac is reset to 0 (or user-specified accrued_interest) because
        # POF_PRD will compute accrual from SD to PRD. Setting ipac to the accrual
        # from last event to SD would double-count interest.
        if ied < sd and sd < prd:
            ipac = jnp.array(attrs.accrued_interest or 0.0, dtype=jnp.float32)
            state = state.replace(sd=sd, ipac=ipac)

        return state

    def initialize_state(self) -> ContractState:
        """Initialize LAM contract state.

        When IED < SD (contract already existed), state is initialized
        as if STF_IED already ran: Nt, Ipnr are set from attributes,
        and interest is accrued from IED (or IPANX) to SD.

        Returns:
            Initial contract state
        """
        attrs = self.attributes
        sd = attrs.status_date
        ied = attrs.initial_exchange_date
        role_sign = contract_role_sign(attrs.contract_role)

        # Initialize Prnxt (next principal redemption amount)
        if attrs.next_principal_redemption_amount is not None:
            prnxt_val = attrs.next_principal_redemption_amount
        elif (
            attrs.notional_principal
            and attrs.principal_redemption_cycle
            and attrs.maturity_date
            and ied
        ):
            # Auto-calculate: PRNXT = NT / number_of_PR_periods
            pr_anchor = attrs.principal_redemption_anchor or ied
            pr_dates = generate_schedule(
                start=pr_anchor,
                cycle=attrs.principal_redemption_cycle,
                end=attrs.maturity_date,
            )
            pr_dates = [d for d in pr_dates if d <= attrs.maturity_date]
            if attrs.maturity_date not in pr_dates:
                pr_dates.append(attrs.maturity_date)
            num_periods = len(pr_dates)
            prnxt_val = attrs.notional_principal / num_periods if num_periods > 0 else 0.0
        else:
            prnxt_val = 0.0
        prnxt = jnp.array(role_sign * prnxt_val, dtype=jnp.float32)

        # PRD pre-simulation: run events from IED to PRD to get correct state
        if attrs.purchase_date and ied:
            return self._pre_simulate_to_prd(attrs, prnxt)

        needs_post_ied = ied and ied < sd
        if needs_post_ied:
            # Contract already started - initialize post-IED state
            nt = role_sign * (attrs.notional_principal or 0.0)
            ipnr = attrs.nominal_interest_rate or 0.0
            dcc = attrs.day_count_convention or DayCountConvention.A360

            init_sd = sd
            accrual_start = attrs.interest_payment_anchor or ied
            if attrs.accrued_interest is not None:
                ipac = attrs.accrued_interest
            elif accrual_start and accrual_start < sd:
                yf = year_fraction(accrual_start, sd, dcc)
                ipac = yf * ipnr * nt
            else:
                ipac = 0.0

            if attrs.interest_calculation_base_amount is not None:
                ipcb_val = role_sign * attrs.interest_calculation_base_amount
            else:
                ipcb_val = nt  # IPCB initialized to signed notional at IED
            return ContractState(
                sd=init_sd,
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

    def get_payoff_function(self, event_type: Any) -> LAMPayoffFunction:
        """Get LAM payoff function.

        Args:
            event_type: Type of event (not used, all events use same POF)

        Returns:
            LAM payoff function instance
        """
        return LAMPayoffFunction(
            contract_role=self.attributes.contract_role,
            currency=self.attributes.currency,
            settlement_currency=None,
        )

    def get_state_transition_function(self, event_type: Any) -> LAMStateTransitionFunction:
        """Get LAM state transition function.

        Args:
            event_type: Type of event (not used, all events use same STF)

        Returns:
            LAM state transition function instance
        """
        return LAMStateTransitionFunction()
