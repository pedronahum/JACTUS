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
    EventSchedule,
    EventType,
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

    def calculate_payoff(
        self,
        event_type: Any,
        state: ContractState,
        attributes: ContractAttributes,
        time: ActusDateTime,
        risk_factor_observer: RiskFactorObserver,
        child_contract_observer: Any | None = None,
    ) -> jnp.ndarray:
        """Calculate payoff for NAM event.

        Args:
            event_type: Type of event
            state: Contract state before event
            attributes: Contract attributes
            time: Event time
            risk_factor_observer: Risk factor observer

        Returns:
            JAX array containing the payoff amount
        """
        if event_type == EventType.AD:
            return self._pof_ad(state, attributes)
        if event_type == EventType.IED:
            return self._pof_ied(state, attributes)
        if event_type == EventType.PR:
            return self._pof_pr(state, attributes, time)
        if event_type == EventType.MD:
            return self._pof_md(state, attributes)
        if event_type == EventType.PP:
            return self._pof_pp(state, attributes, time, risk_factor_observer)
        if event_type == EventType.PY:
            return self._pof_py(state, attributes, time, risk_factor_observer)
        if event_type == EventType.FP:
            return self._pof_fp(state, attributes, time, risk_factor_observer)
        if event_type == EventType.PRD:
            return self._pof_prd(state, attributes, time)
        if event_type == EventType.TD:
            return self._pof_td(state, attributes, time)
        if event_type == EventType.IP:
            return self._pof_ip(state, attributes, time)
        if event_type == EventType.IPCI:
            return self._pof_ipci(state, attributes)
        if event_type == EventType.IPCB:
            return self._pof_ipcb(state, attributes)
        if event_type == EventType.RR:
            return self._pof_rr(state, attributes)
        if event_type == EventType.RRF:
            return self._pof_rrf(state, attributes)
        if event_type == EventType.SC:
            return self._pof_sc(state, attributes)
        if event_type == EventType.CE:
            return self._pof_ce(state, attributes)
        return jnp.array(0.0, dtype=jnp.float32)

    def _pof_ad(self, state: ContractState, attrs: ContractAttributes) -> jnp.ndarray:
        """POF_AD: Analysis Date - no cashflow."""
        return jnp.array(0.0, dtype=jnp.float32)

    def _pof_ied(self, state: ContractState, attrs: ContractAttributes) -> jnp.ndarray:
        """POF_IED: Initial Exchange - disburse principal.

        Formula: R(CNTRL) × (-1) × NT
        """
        role_sign = contract_role_sign(self.contract_role)
        return role_sign * jnp.array(-1.0, dtype=jnp.float32) * state.nsc * attrs.notional_principal

    def _pof_pr(
        self, state: ContractState, attrs: ContractAttributes, time: ActusDateTime
    ) -> jnp.ndarray:
        """POF_PR: Principal Redemption - pay principal NET of accrued interest.

        Formula: R(CNTRL) × Nsc × (Prnxt - Ipac - Y(Sd, t) × Ipnr × Ipcb)

        Key Feature: Payoff can be NEGATIVE if interest > payment.
        This represents the borrower receiving money (negative amortization).
        """
        role_sign = contract_role_sign(self.contract_role)
        yf = year_fraction(state.sd, time, attrs.day_count_convention or attrs.day_count_convention)

        # Calculate accrued interest using IPCB
        ipcb = state.ipcb if state.ipcb is not None else state.nt
        accrued_interest = yf * state.ipnr * ipcb

        # Net payment = Prnxt - already accrued - newly accrued
        # If this is negative, the notional increases (negative amortization)
        net_payment = state.prnxt - state.ipac - accrued_interest

        return role_sign * state.nsc * net_payment

    def _pof_md(self, state: ContractState, attrs: ContractAttributes) -> jnp.ndarray:
        """POF_MD: Maturity - final principal + accrued interest + fees.

        Formula: R(CNTRL) × (Nsc × Nt + Isc × Ipac + Feac)
        """
        role_sign = contract_role_sign(self.contract_role)
        return role_sign * (state.nsc * state.nt + state.isc * state.ipac + state.feac)

    def _pof_pp(
        self,
        state: ContractState,
        attrs: ContractAttributes,
        time: ActusDateTime,
        rf_obs: RiskFactorObserver,
    ) -> jnp.ndarray:
        """POF_PP: Prepayment - observed prepayment amount."""
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

        Formula: R(CNTRL) × Feac
        """
        role_sign = contract_role_sign(self.contract_role)
        return role_sign * state.feac

    def _pof_prd(
        self, state: ContractState, attrs: ContractAttributes, time: ActusDateTime
    ) -> jnp.ndarray:
        """POF_PRD: Purchase - purchase price + accrued interest on IPCB.

        Formula: R(CNTRL) × (-1) × (PPRD + Ipac + Y(Sd, t) × Ipnr × Ipcb)
        """
        role_sign = contract_role_sign(self.contract_role)
        yf = year_fraction(state.sd, time, attrs.day_count_convention or attrs.day_count_convention)

        ipcb = state.ipcb if state.ipcb is not None else state.nt
        accrued_interest = yf * state.ipnr * ipcb

        pprd = attrs.price_at_purchase_date or 0.0
        return (
            role_sign
            * jnp.array(-1.0, dtype=jnp.float32)
            * (jnp.array(pprd, dtype=jnp.float32) + state.ipac + accrued_interest)
        )

    def _pof_td(
        self, state: ContractState, attrs: ContractAttributes, time: ActusDateTime
    ) -> jnp.ndarray:
        """POF_TD: Termination - termination price + accrued interest on IPCB.

        Formula: R(CNTRL) × (PTD + Ipac + Y(Sd, t) × Ipnr × Ipcb)
        """
        role_sign = contract_role_sign(self.contract_role)
        yf = year_fraction(state.sd, time, attrs.day_count_convention or attrs.day_count_convention)

        ipcb = state.ipcb if state.ipcb is not None else state.nt
        accrued_interest = yf * state.ipnr * ipcb

        ptd = attrs.price_at_termination_date or 0.0
        return role_sign * (jnp.array(ptd, dtype=jnp.float32) + state.ipac + accrued_interest)

    def _pof_ip(
        self, state: ContractState, attrs: ContractAttributes, time: ActusDateTime
    ) -> jnp.ndarray:
        """POF_IP: Interest Payment - accrued interest on IPCB.

        Formula: R(CNTRL) × Isc × (Ipac + Y(Sd, t) × Ipnr × Ipcb)
        """
        role_sign = contract_role_sign(self.contract_role)
        yf = year_fraction(state.sd, time, attrs.day_count_convention or attrs.day_count_convention)

        ipcb = state.ipcb if state.ipcb is not None else state.nt
        accrued_interest = yf * state.ipnr * ipcb

        return role_sign * state.isc * (state.ipac + accrued_interest)

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

    def transition_state(
        self,
        event_type: Any,
        state: ContractState,
        attributes: ContractAttributes,
        time: ActusDateTime,
        risk_factor_observer: RiskFactorObserver,
        child_contract_observer: Any | None = None,
    ) -> ContractState:
        """Apply state transition for NAM event.

        Args:
            event_type: Type of event
            state: Contract state before event
            attributes: Contract attributes
            time: Event time
            risk_factor_observer: Observer for market data

        Returns:
            New contract state after event
        """
        if event_type == EventType.AD:
            return self._stf_ad(state, attributes, time, risk_factor_observer)
        if event_type == EventType.IED:
            return self._stf_ied(state, attributes, time, risk_factor_observer)
        if event_type == EventType.PR:
            return self._stf_pr(state, attributes, time, risk_factor_observer)
        if event_type == EventType.MD:
            return self._stf_md(state, attributes, time, risk_factor_observer)
        if event_type == EventType.PP:
            return self._stf_pp(state, attributes, time, risk_factor_observer)
        if event_type == EventType.PY:
            return self._stf_py(state, attributes, time, risk_factor_observer)
        if event_type == EventType.FP:
            return self._stf_fp(state, attributes, time, risk_factor_observer)
        if event_type == EventType.PRD:
            return self._stf_prd(state, attributes, time, risk_factor_observer)
        if event_type == EventType.TD:
            return self._stf_td(state, attributes, time, risk_factor_observer)
        if event_type == EventType.IP:
            return self._stf_ip(state, attributes, time, risk_factor_observer)
        if event_type == EventType.IPCI:
            return self._stf_ipci(state, attributes, time, risk_factor_observer)
        if event_type == EventType.IPCB:
            return self._stf_ipcb(state, attributes, time, risk_factor_observer)
        if event_type == EventType.RR:
            return self._stf_rr(state, attributes, time, risk_factor_observer)
        if event_type == EventType.RRF:
            return self._stf_rrf(state, attributes, time, risk_factor_observer)
        if event_type == EventType.SC:
            return self._stf_sc(state, attributes, time, risk_factor_observer)
        if event_type == EventType.CE:
            return self._stf_ce(state, attributes, time, risk_factor_observer)
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

        # Initialize prnxt
        prnxt = jnp.array(attrs.next_principal_redemption_amount or 0.0, dtype=jnp.float32)

        return state.replace(
            sd=time,
            nt=role_sign * jnp.array(attrs.notional_principal, dtype=jnp.float32),
            ipnr=jnp.array(attrs.nominal_interest_rate or 0.0, dtype=jnp.float32),
            ipac=jnp.array(0.0, dtype=jnp.float32),
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
        role_sign = contract_role_sign(attrs.contract_role)
        yf = year_fraction(state.sd, time, attrs.day_count_convention or attrs.day_count_convention)

        # Calculate accrued interest using current IPCB
        ipcb = state.ipcb if state.ipcb is not None else state.nt
        accrued_interest = yf * state.ipnr * ipcb

        # Net principal reduction = payment - all interest
        # If negative, this represents principal increase
        net_principal_reduction = state.prnxt - state.ipac - accrued_interest

        # Update notional (can increase if net_principal_reduction < 0)
        new_nt = state.nt - role_sign * net_principal_reduction

        # Update IPCB if mode is 'NT'
        ipcb_mode = attrs.interest_calculation_base or "NT"
        if ipcb_mode == "NT":
            new_ipcb = new_nt
        elif ipcb_mode == "NTIED":
            new_ipcb = state.ipcb  # Fixed at IED
        else:  # NTL
            new_ipcb = state.ipcb  # Only updated at IPCB events

        return state.replace(
            sd=time,
            nt=new_nt,
            ipac=jnp.array(0.0, dtype=jnp.float32),  # Reset to zero (all interest handled)
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
        """STF_PP: Prepayment - reduce notional, update IPCB if needed."""
        return state.replace(sd=time)

    def _stf_py(
        self,
        state: ContractState,
        attrs: ContractAttributes,
        time: ActusDateTime,
        risk_factor_observer: RiskFactorObserver,
        child_contract_observer: Any | None = None,
    ) -> ContractState:
        """STF_PY: Penalty - accrue interest and fees."""
        yf = year_fraction(state.sd, time, attrs.day_count_convention or attrs.day_count_convention)
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
        yf = year_fraction(state.sd, time, attrs.day_count_convention or attrs.day_count_convention)
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
        """STF_PRD: Purchase - update status date."""
        return state.replace(sd=time)

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
        yf = year_fraction(state.sd, time, attrs.day_count_convention or attrs.day_count_convention)
        ipcb = state.ipcb if state.ipcb is not None else state.nt
        accrued = yf * state.ipnr * ipcb

        new_nt = state.nt + state.ipac + accrued

        # Update IPCB if mode is 'NT'
        ipcb_mode = attrs.interest_calculation_base or "NT"
        new_ipcb = new_nt if ipcb_mode == "NT" else state.ipcb

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
        yf = year_fraction(state.sd, time, attrs.day_count_convention or attrs.day_count_convention)
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
        """STF_RR: Rate Reset - update interest rate."""
        yf = year_fraction(state.sd, time, attrs.day_count_convention or attrs.day_count_convention)
        ipcb = state.ipcb if state.ipcb is not None else state.nt
        new_ipac = state.ipac + yf * state.ipnr * ipcb

        return state.replace(sd=time, ipac=new_ipac)

    def _stf_rrf(
        self,
        state: ContractState,
        attrs: ContractAttributes,
        time: ActusDateTime,
        risk_factor_observer: RiskFactorObserver,
        child_contract_observer: Any | None = None,
    ) -> ContractState:
        """STF_RRF: Rate Reset Fixing - fix interest rate."""
        yf = year_fraction(state.sd, time, attrs.day_count_convention or attrs.day_count_convention)
        ipcb = state.ipcb if state.ipcb is not None else state.nt
        new_ipac = state.ipac + yf * state.ipnr * ipcb

        return state.replace(sd=time, ipac=new_ipac)

    def _stf_sc(
        self,
        state: ContractState,
        attrs: ContractAttributes,
        time: ActusDateTime,
        risk_factor_observer: RiskFactorObserver,
        child_contract_observer: Any | None = None,
    ) -> ContractState:
        """STF_SC: Scaling - update scaling multipliers."""
        yf = year_fraction(state.sd, time, attrs.day_count_convention or attrs.day_count_convention)
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
        """Generate NAM event schedule.

        NAM schedule is same as LAM, except:
        - IP schedule ends one period before PR schedule (per ACTUS spec)

        Returns:
            Event schedule
        """
        events: list[ContractEvent] = []

        # 1. Initial Exchange Date
        if self.attributes.initial_exchange_date:
            events.append(
                ContractEvent(
                    event_type=EventType.IED,
                    event_time=self.attributes.initial_exchange_date,
                    payoff=jnp.array(0.0, dtype=jnp.float32),
                    currency=self.attributes.currency or "XXX",
                    sequence=len(events),
                )
            )

        # 2. Principal Redemption schedule
        pr_events_times = []
        if self.attributes.principal_redemption_cycle:
            pr_schedule = generate_schedule(
                start=self.attributes.principal_redemption_anchor
                or self.attributes.initial_exchange_date,
                cycle=self.attributes.principal_redemption_cycle,
                end=self.attributes.maturity_date,
            )

            for pr_date in pr_schedule:
                # Don't include maturity date in PR schedule
                if self.attributes.maturity_date and pr_date >= self.attributes.maturity_date:
                    break
                if pr_date > self.attributes.initial_exchange_date:
                    pr_events_times.append(pr_date)
                    events.append(
                        ContractEvent(
                            event_type=EventType.PR,
                            event_time=pr_date,
                            payoff=jnp.array(0.0, dtype=jnp.float32),
                            currency=self.attributes.currency or "XXX",
                            sequence=len(events),
                        )
                    )

        # 3. Interest Payment schedule (ends one period before PR schedule for NAM)
        if self.attributes.interest_payment_cycle:
            ip_schedule = generate_schedule(
                start=self.attributes.interest_payment_anchor
                or self.attributes.initial_exchange_date,
                cycle=self.attributes.interest_payment_cycle,
                end=self.attributes.maturity_date,
            )

            # For NAM: IP schedule ends one period before last PR event
            # Find the second-to-last PR event time
            ip_end_time = None
            if len(pr_events_times) > 1:
                ip_end_time = pr_events_times[-2]  # Second to last PR event
            elif self.attributes.maturity_date:
                ip_end_time = self.attributes.maturity_date

            for ip_date in ip_schedule:
                # Stop IP schedule before the last PR period
                if ip_end_time and ip_date >= ip_end_time:
                    break
                if ip_date > self.attributes.initial_exchange_date:
                    events.append(
                        ContractEvent(
                            event_type=EventType.IP,
                            event_time=ip_date,
                            payoff=jnp.array(0.0, dtype=jnp.float32),
                            currency=self.attributes.currency or "XXX",
                            sequence=len(events),
                        )
                    )

        # 4. IPCB schedule (only if IPCB='NTL')
        if (
            self.attributes.interest_calculation_base == "NTL"
            and self.attributes.interest_calculation_base_cycle
        ):
            ipcb_schedule = generate_schedule(
                start=self.attributes.interest_calculation_base_anchor
                or self.attributes.initial_exchange_date,
                cycle=self.attributes.interest_calculation_base_cycle,
                end=self.attributes.maturity_date,
            )

            for ipcb_date in ipcb_schedule:
                if ipcb_date > self.attributes.initial_exchange_date:
                    events.append(
                        ContractEvent(
                            event_type=EventType.IPCB,
                            event_time=ipcb_date,
                            payoff=jnp.array(0.0, dtype=jnp.float32),
                            currency=self.attributes.currency or "XXX",
                            sequence=len(events),
                        )
                    )

        # 5. Maturity Date (if defined)
        if self.attributes.maturity_date:
            events.append(
                ContractEvent(
                    event_type=EventType.MD,
                    event_time=self.attributes.maturity_date,
                    payoff=jnp.array(0.0, dtype=jnp.float32),
                    currency=self.attributes.currency or "XXX",
                    sequence=len(events),
                )
            )

        # Sort by time and reassign sequence numbers
        events.sort(key=lambda e: (e.event_time.to_iso(), e.sequence))

        # Reassign sequence numbers after sorting
        for i in range(len(events)):
            events[i] = ContractEvent(
                event_type=events[i].event_type,
                event_time=events[i].event_time,
                payoff=events[i].payoff,
                currency=events[i].currency,
                sequence=i,
            )

        return EventSchedule(events=events, contract_id=self.attributes.contract_id)

    def initialize_state(self) -> ContractState:
        """Initialize NAM contract state.

        Same as LAM: Initialize Prnxt and Ipcb.

        Returns:
            Initial contract state
        """
        role_sign = contract_role_sign(self.attributes.contract_role)

        # Initialize Prnxt (next principal redemption amount)
        prnxt = jnp.array(
            role_sign * self.attributes.next_principal_redemption_amount,
            dtype=jnp.float32,
        )

        return ContractState(
            sd=self.attributes.status_date,
            tmd=self.attributes.maturity_date,
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
