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
            return handler(state, attributes, time, risk_factor_observer)
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

    def _pof_pr(self, state: ContractState, attrs: ContractAttributes) -> jnp.ndarray:
        """POF_PR: Principal Redemption - pay fixed principal amount.

        Formula: R(CNTRL) × Nsc × Prnxt

        Key Feature: Uses Prnxt (next principal redemption amount) from state.
        """
        role_sign = contract_role_sign(self.contract_role)
        return role_sign * state.nsc * state.prnxt

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
        """POF_PP: Prepayment - observed prepayment amount.

        Note: Amount from risk factor observer, not yet implemented.
        """
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

        Formula: R(CNTRL) × Feac
        """
        role_sign = contract_role_sign(self.contract_role)
        return role_sign * state.feac

    def _pof_prd(
        self, state: ContractState, attrs: ContractAttributes, time: ActusDateTime
    ) -> jnp.ndarray:
        """POF_PRD: Purchase - purchase price + accrued interest on IPCB.

        Formula: R(CNTRL) × (-1) × (PPRD + Ipac + Y(Sd, t) × Ipnr × Ipcb)

        Key Feature: Uses Ipcb instead of Nt for interest calculation.
        """
        role_sign = contract_role_sign(self.contract_role)
        yf = year_fraction(state.sd, time, attrs.day_count_convention or attrs.day_count_convention)

        # Use IPCB for interest calculation
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

        Key Feature: Uses Ipcb instead of Nt for interest calculation.
        """
        role_sign = contract_role_sign(self.contract_role)
        yf = year_fraction(state.sd, time, attrs.day_count_convention or attrs.day_count_convention)

        # Use IPCB for interest calculation
        ipcb = state.ipcb if state.ipcb is not None else state.nt
        accrued_interest = yf * state.ipnr * ipcb

        ptd = attrs.price_at_termination_date or 0.0
        return role_sign * (jnp.array(ptd, dtype=jnp.float32) + state.ipac + accrued_interest)

    def _pof_ip(
        self, state: ContractState, attrs: ContractAttributes, time: ActusDateTime
    ) -> jnp.ndarray:
        """POF_IP: Interest Payment - accrued interest on IPCB.

        Formula: R(CNTRL) × Isc × (Ipac + Y(Sd, t) × Ipnr × Ipcb)

        Key Feature: Interest calculated on Ipcb, NOT current notional.
        """
        role_sign = contract_role_sign(self.contract_role)
        yf = year_fraction(state.sd, time, attrs.day_count_convention or attrs.day_count_convention)

        # Use IPCB for interest calculation
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
            return handler(state, attributes, time, risk_factor_observer)
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
    ) -> ContractState:
        """STF_PR: Principal Redemption - reduce notional, update IPCB if needed.

        Formula:
            Nt = Nt - R(CNTRL) × Prnxt
            Ipac = Ipac + Y(Sd, t) × Ipnr × Ipcb
            Ipcb = Nt (if IPCB='NT')

        Key Feature: Update IPCB to new notional if mode is 'NT'.
        """
        role_sign = contract_role_sign(attrs.contract_role)
        yf = year_fraction(state.sd, time, attrs.day_count_convention or attrs.day_count_convention)

        # Calculate accrued interest using current IPCB
        ipcb = state.ipcb if state.ipcb is not None else state.nt
        new_ipac = state.ipac + yf * state.ipnr * ipcb

        # Reduce notional by fixed amount
        new_nt = state.nt - role_sign * state.prnxt

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
        """STF_PP: Prepayment - reduce notional, update IPCB if needed.

        Note: Not yet fully implemented.
        """
        return state.replace(sd=time)

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
    ) -> ContractState:
        """STF_FP: Fee Payment - reset accrued fees.

        Note: Not yet fully implemented.
        """
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
    ) -> ContractState:
        """STF_PRD: Purchase - update status date."""
        return state.replace(sd=time)

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
        yf = year_fraction(state.sd, time, attrs.day_count_convention or attrs.day_count_convention)
        ipcb = state.ipcb if state.ipcb is not None else state.nt
        accrued = yf * state.ipnr * ipcb

        new_nt = state.nt + state.ipac + accrued

        # Update IPCB if mode is 'NT'
        ipcb_mode = attrs.interest_calculation_base or "NT"
        if ipcb_mode == "NT":
            new_ipcb = new_nt
        else:
            new_ipcb = state.ipcb

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
    ) -> ContractState:
        """STF_RR: Rate Reset - update interest rate.

        Note: Not yet fully implemented.
        """
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
    ) -> ContractState:
        """STF_RRF: Rate Reset Fixing - fix interest rate.

        Note: Not yet fully implemented.
        """
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
    ) -> ContractState:
        """STF_SC: Scaling - update scaling multipliers.

        Note: Not yet fully implemented.
        """
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
        """Generate LAM event schedule.

        LAM schedule includes:
        - IED: Initial exchange
        - PR: Principal redemption events (regular cycle)
        - IP: Interest payment events (if cycle defined)
        - IPCB: Interest calculation base fixing (if IPCB='NTL')
        - MD: Maturity (if exists after all PR events)

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
                    events.append(
                        ContractEvent(
                            event_type=EventType.PR,
                            event_time=pr_date,
                            payoff=jnp.array(0.0, dtype=jnp.float32),
                            currency=self.attributes.currency or "XXX",
                            sequence=len(events),
                        )
                    )

        # 3. Interest Payment schedule (if defined)
        if self.attributes.interest_payment_cycle:
            ip_schedule = generate_schedule(
                start=self.attributes.interest_payment_anchor
                or self.attributes.initial_exchange_date,
                cycle=self.attributes.interest_payment_cycle,
                end=self.attributes.maturity_date,
            )

            for ip_date in ip_schedule:
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
        if self.attributes.interest_calculation_base == "NTL":
            if self.attributes.interest_calculation_base_cycle:
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
        """Initialize LAM contract state.

        Initializes all state variables including Prnxt and Ipcb.

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
