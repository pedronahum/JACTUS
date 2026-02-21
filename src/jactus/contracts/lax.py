"""Exotic Linear Amortizer (LAX) contract implementation.

This module implements the LAX contract type - the most complex amortizing contract
with flexible array schedules that allow varying principal redemption amounts, rates,
and cycles over the life of the contract.

ACTUS Reference:
    ACTUS v1.1 Section 7.3 - LAX: Exotic Linear Amortizer

Key Features:
    - Array schedules for principal redemption (ARPRANX, ARPRCL, ARPRNXT)
    - Array schedules for interest payments (ARIPANX, ARIPCL)
    - Array schedules for rate resets (ARRRANX, ARRRCL, ARRATE)
    - Increase/decrease indicators (ARINCDEC) for principal changes
    - Fixed/variable rate indicators (ARFIXVAR)
    - PI (Principal Increase) and PR (Principal Redemption) events
    - PRF (Principal Redemption Amount Fixing) events
    - All IPCB modes from LAM

Array Schedule Concept:
    Instead of a single cycle and anchor, LAX uses arrays to define multiple
    sub-schedules with different parameters. For example:
    - ARPRANX = [2024-01-15, 2025-01-15, 2026-01-15]
    - ARPRCL = ["1M", "1M", "1M"]
    - ARPRNXT = [1000, 2000, 3000]
    - ARINCDEC = ["INC", "INC", "DEC"]
    This creates increasing principal for 2 years, then decreasing.

Example:
    >>> from jactus.contracts import create_contract
    >>> from jactus.core import ContractAttributes, ContractType, ContractRole
    >>> from jactus.core import ActusDateTime, DayCountConvention
    >>> from jactus.observers import ConstantRiskFactorObserver
    >>>
    >>> attrs = ContractAttributes(
    ...     contract_id="STEP-UP-LOAN-001",
    ...     contract_type=ContractType.LAX,
    ...     contract_role=ContractRole.RPA,
    ...     status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
    ...     initial_exchange_date=ActusDateTime(2024, 1, 15, 0, 0, 0),
    ...     maturity_date=ActusDateTime(2027, 1, 15, 0, 0, 0),
    ...     currency="USD",
    ...     notional_principal=100000.0,
    ...     nominal_interest_rate=0.05,
    ...     day_count_convention=DayCountConvention.A360,
    ...     array_pr_anchor=[ActusDateTime(2024, 2, 15), ActusDateTime(2025, 1, 15)],
    ...     array_pr_cycle=["1M", "1M"],
    ...     array_pr_next=[1000.0, 2000.0],
    ...     array_increase_decrease=["INC", "DEC"]
    ... )
    >>>
    >>> rf_obs = ConstantRiskFactorObserver(constant_value=0.05)
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


def generate_array_schedule(
    anchors: list[ActusDateTime],
    cycles: list[str],
    end: ActusDateTime,
    filter_values: list[str] | None = None,
    filter_target: str | None = None,
) -> list[ActusDateTime]:
    """Generate schedule from array of anchors and cycles.

    Args:
        anchors: Array of anchor dates (start dates for each sub-schedule)
        cycles: Array of cycles (one per anchor)
        end: End date (maturity date)
        filter_values: Optional array of filter values (e.g., ARINCDEC)
        filter_target: Optional target value to filter for (e.g., "DEC")

    Returns:
        Union of all sub-schedules, sorted and deduplicated

    Example:
        >>> anchors = [ActusDateTime(2024, 1, 15), ActusDateTime(2025, 1, 15)]
        >>> cycles = ["1M", "1M"]
        >>> end = ActusDateTime(2026, 1, 15)
        >>> generate_array_schedule(anchors, cycles, end)
        [ActusDateTime(2024, 2, 15), ActusDateTime(2024, 3, 15), ...]
    """
    if not anchors or not cycles:
        return []

    if len(anchors) != len(cycles):
        raise ValueError(
            f"Anchors and cycles must have same length: {len(anchors)} vs {len(cycles)}"
        )

    if filter_values is not None and len(filter_values) != len(anchors):
        raise ValueError(
            f"Filter values must have same length as anchors: {len(filter_values)} vs {len(anchors)}"
        )

    all_events = []

    for i, (anchor, cycle) in enumerate(zip(anchors, cycles)):
        # Skip if filter doesn't match
        if filter_values is not None and filter_target is not None:
            if filter_values[i] != filter_target:
                continue

        # Generate sub-schedule
        sub_schedule = generate_schedule(start=anchor, cycle=cycle, end=end)

        # Filter to events after anchor
        sub_schedule = [d for d in sub_schedule if d > anchor]

        all_events.extend(sub_schedule)

    # Sort and deduplicate
    all_events = sorted(set(all_events))

    # Filter to events before or at maturity
    all_events = [d for d in all_events if d <= end]

    return all_events


class LAXPayoffFunction(BasePayoffFunction):
    """Payoff function for LAX contracts.

    Extends LAM payoff functions with PI (Principal Increase) and PRF
    (Principal Redemption Amount Fixing) events.

    ACTUS Reference:
        ACTUS v1.1 Section 7.3 - LAX Payoff Functions

    Events:
        All LAM events (AD, IED, PR, MD, PP, PY, FP, PRD, TD, IP, IPCI, IPCB, RR, RRF, SC, CE)
        Plus:
        PI: Principal Increase (negative principal redemption)
        PRF: Principal Redemption Amount Fixing (update Prnxt)
    """

    def __init__(self, contract_role, currency, settlement_currency=None):
        """Initialize LAX payoff function.

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
        if event_type == EventType.PI:
            return self._pof_pi(state, attributes, time)
        if event_type == EventType.MD:
            return self._pof_md(state, attributes, time)
        if event_type == EventType.PP:
            return self._pof_pp(state, attributes, time)
        if event_type == EventType.PY:
            return self._pof_py(state, attributes, time)
        if event_type == EventType.FP:
            return self._pof_fp(state, attributes, time)
        if event_type == EventType.PRD:
            return self._pof_prd(state, attributes, time)
        if event_type == EventType.TD:
            return self._pof_td(state, attributes, time)
        if event_type == EventType.IP:
            return self._pof_ip(state, attributes, time)
        if event_type == EventType.IPCI:
            return self._pof_ipci(state, attributes, time)
        if event_type == EventType.IPCB:
            return self._pof_ipcb(state, attributes, time)
        if event_type == EventType.PRF:
            return self._pof_prf(state, attributes, time)
        if event_type == EventType.RR:
            return self._pof_rr(state, attributes, time)
        if event_type == EventType.RRF:
            return self._pof_rrf(state, attributes, time)
        if event_type == EventType.SC:
            return self._pof_sc(state, attributes, time)
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
        """POF_IED: Initial Exchange - disburse principal."""
        role_sign = contract_role_sign(attrs.contract_role)
        return role_sign * state.nsc * state.nt

    def _pof_pr(
        self, state: ContractState, attrs: ContractAttributes, time: ActusDateTime
    ) -> jnp.ndarray:
        """POF_PR: Principal Redemption - pay fixed principal amount.

        Same as LAM: pay Prnxt amount.
        """
        role_sign = contract_role_sign(attrs.contract_role)
        return role_sign * state.nsc * state.prnxt

    def _pof_pi(
        self, state: ContractState, attrs: ContractAttributes, time: ActusDateTime
    ) -> jnp.ndarray:
        """POF_PI: Principal Increase - receive additional principal (negative PR).

        This is like a negative principal redemption - the borrower receives more money.
        """
        role_sign = contract_role_sign(attrs.contract_role)
        # PI is negative PR - so we flip the sign
        return role_sign * state.nsc * state.prnxt

    def _pof_md(
        self, state: ContractState, attrs: ContractAttributes, time: ActusDateTime
    ) -> jnp.ndarray:
        """POF_MD: Maturity - pay remaining principal and accrued interest."""
        role_sign = contract_role_sign(attrs.contract_role)
        yf = year_fraction(state.sd, time, attrs.day_count_convention or attrs.day_count_convention)
        ipcb = state.ipcb if state.ipcb is not None else state.nt
        accrued = yf * state.ipnr * ipcb
        return role_sign * state.nsc * (state.nt + state.ipac + accrued)

    def _pof_pp(
        self, state: ContractState, attrs: ContractAttributes, time: ActusDateTime
    ) -> jnp.ndarray:
        """POF_PP: Prepayment - not yet implemented."""
        return jnp.array(0.0, dtype=jnp.float32)

    def _pof_py(
        self, state: ContractState, attrs: ContractAttributes, time: ActusDateTime
    ) -> jnp.ndarray:
        """POF_PY: Penalty - not yet implemented."""
        return jnp.array(0.0, dtype=jnp.float32)

    def _pof_fp(
        self, state: ContractState, attrs: ContractAttributes, time: ActusDateTime
    ) -> jnp.ndarray:
        """POF_FP: Fee Payment - pay accrued fees."""
        role_sign = contract_role_sign(attrs.contract_role)
        return role_sign * state.feac

    def _pof_prd(
        self, state: ContractState, attrs: ContractAttributes, time: ActusDateTime
    ) -> jnp.ndarray:
        """POF_PRD: Purchase - not yet implemented."""
        return jnp.array(0.0, dtype=jnp.float32)

    def _pof_td(
        self, state: ContractState, attrs: ContractAttributes, time: ActusDateTime
    ) -> jnp.ndarray:
        """POF_TD: Termination - pay notional and accrued interest."""
        role_sign = contract_role_sign(attrs.contract_role)
        yf = year_fraction(state.sd, time, attrs.day_count_convention or attrs.day_count_convention)
        ipcb = state.ipcb if state.ipcb is not None else state.nt
        accrued = yf * state.ipnr * ipcb
        return role_sign * state.nsc * (state.nt + state.ipac + accrued)

    def _pof_ip(
        self, state: ContractState, attrs: ContractAttributes, time: ActusDateTime
    ) -> jnp.ndarray:
        """POF_IP: Interest Payment - pay accrued interest on IPCB."""
        role_sign = contract_role_sign(attrs.contract_role)
        yf = year_fraction(state.sd, time, attrs.day_count_convention or attrs.day_count_convention)
        ipcb = state.ipcb if state.ipcb is not None else state.nt
        accrued = yf * state.ipnr * ipcb
        return role_sign * state.isc * (state.ipac + accrued)

    def _pof_ipci(
        self, state: ContractState, attrs: ContractAttributes, time: ActusDateTime
    ) -> jnp.ndarray:
        """POF_IPCI: Interest Capitalization - no payoff."""
        return jnp.array(0.0, dtype=jnp.float32)

    def _pof_ipcb(
        self, state: ContractState, attrs: ContractAttributes, time: ActusDateTime
    ) -> jnp.ndarray:
        """POF_IPCB: Interest Calculation Base Fixing - no payoff."""
        return jnp.array(0.0, dtype=jnp.float32)

    def _pof_prf(
        self, state: ContractState, attrs: ContractAttributes, time: ActusDateTime
    ) -> jnp.ndarray:
        """POF_PRF: Principal Redemption Amount Fixing - no payoff.

        PRF events update Prnxt but don't generate cashflows.
        """
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

    def _pof_sc(
        self, state: ContractState, attrs: ContractAttributes, time: ActusDateTime
    ) -> jnp.ndarray:
        """POF_SC: Scaling - no payoff."""
        return jnp.array(0.0, dtype=jnp.float32)

    def _pof_ce(
        self, state: ContractState, attrs: ContractAttributes, time: ActusDateTime
    ) -> jnp.ndarray:
        """POF_CE: Credit Event - not yet implemented."""
        return jnp.array(0.0, dtype=jnp.float32)


class LAXStateTransitionFunction(BaseStateTransitionFunction):
    """State transition function for LAX contracts.

    Extends LAM state transitions with PI (Principal Increase) and PRF
    (Principal Redemption Amount Fixing) events.

    ACTUS Reference:
        ACTUS v1.1 Section 7.3 - LAX State Transition Functions

    Key Differences from LAM:
        - PI events: Increase notional (opposite of PR)
        - PRF events: Fix Prnxt from array schedule
        - Array-based schedule generation
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
        if event_type == EventType.PI:
            return self._stf_pi(state, attributes, time, risk_factor_observer)
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
        if event_type == EventType.PRF:
            return self._stf_prf(state, attributes, time, risk_factor_observer)
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

        Same as LAM initialization.
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

        # Initialize prnxt - for LAX this will be updated at PRF events
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

        Same as LAM: reduce notional by Prnxt amount.
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

    def _stf_pi(
        self,
        state: ContractState,
        attrs: ContractAttributes,
        time: ActusDateTime,
        risk_factor_observer: RiskFactorObserver,
    ) -> ContractState:
        """STF_PI: Principal Increase - increase notional, update IPCB if needed.

        This is the opposite of PR - the notional increases by Prnxt.
        """
        role_sign = contract_role_sign(attrs.contract_role)
        yf = year_fraction(state.sd, time, attrs.day_count_convention or attrs.day_count_convention)

        # Calculate accrued interest using current IPCB
        ipcb = state.ipcb if state.ipcb is not None else state.nt
        new_ipac = state.ipac + yf * state.ipnr * ipcb

        # Increase notional by fixed amount (opposite of PR)
        new_nt = state.nt + role_sign * state.prnxt

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
        """STF_PP: Prepayment - not yet implemented."""
        return state.replace(sd=time)

    def _stf_py(
        self,
        state: ContractState,
        attrs: ContractAttributes,
        time: ActusDateTime,
        risk_factor_observer: RiskFactorObserver,
    ) -> ContractState:
        """STF_PY: Penalty - not yet implemented."""
        return state.replace(sd=time)

    def _stf_fp(
        self,
        state: ContractState,
        attrs: ContractAttributes,
        time: ActusDateTime,
        risk_factor_observer: RiskFactorObserver,
    ) -> ContractState:
        """STF_FP: Fee Payment - reset accrued fees."""
        yf = year_fraction(state.sd, time, attrs.day_count_convention or attrs.day_count_convention)

        # Accrue any remaining fees
        if attrs.fee_rate and attrs.fee_basis:
            # Simplified fee accrual on notional
            new_feac = state.feac + yf * attrs.fee_rate * abs(state.nt)
        else:
            new_feac = state.feac

        # Reset fees after payment
        return state.replace(sd=time, feac=jnp.array(0.0, dtype=jnp.float32))

    def _stf_prd(
        self,
        state: ContractState,
        attrs: ContractAttributes,
        time: ActusDateTime,
        risk_factor_observer: RiskFactorObserver,
    ) -> ContractState:
        """STF_PRD: Purchase - not yet implemented."""
        return state.replace(sd=time)

    def _stf_td(
        self,
        state: ContractState,
        attrs: ContractAttributes,
        time: ActusDateTime,
        risk_factor_observer: RiskFactorObserver,
    ) -> ContractState:
        """STF_TD: Termination - zero out all state variables."""
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
        yf = year_fraction(state.sd, time, attrs.day_count_convention or attrs.day_count_convention)
        ipcb = state.ipcb if state.ipcb is not None else state.nt
        accrued = yf * state.ipnr * ipcb

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
        role_sign = contract_role_sign(attrs.contract_role)
        yf = year_fraction(state.sd, time, attrs.day_count_convention or attrs.day_count_convention)
        ipcb = state.ipcb if state.ipcb is not None else state.nt
        accrued = yf * state.ipnr * ipcb

        # Add accrued interest to notional
        new_nt = state.nt + role_sign * (state.ipac + accrued)

        # Update IPCB if mode is 'NT'
        ipcb_mode = attrs.interest_calculation_base or "NT"
        if ipcb_mode == "NT":
            new_ipcb = new_nt
        else:
            new_ipcb = state.ipcb

        return state.replace(
            sd=time,
            nt=new_nt,
            ipac=jnp.array(0.0, dtype=jnp.float32),
            ipcb=new_ipcb,
        )

    def _stf_ipcb(
        self,
        state: ContractState,
        attrs: ContractAttributes,
        time: ActusDateTime,
        risk_factor_observer: RiskFactorObserver,
    ) -> ContractState:
        """STF_IPCB: Interest Calculation Base Fixing - update IPCB to current notional.

        Only applies when IPCB mode is 'NTL' (lagged notional).
        """
        ipcb_mode = attrs.interest_calculation_base or "NT"
        if ipcb_mode == "NTL":
            # Fix IPCB to current notional
            new_ipcb = state.nt
        else:
            new_ipcb = state.ipcb

        return state.replace(sd=time, ipcb=new_ipcb)

    def _stf_prf(
        self,
        state: ContractState,
        attrs: ContractAttributes,
        time: ActusDateTime,
        risk_factor_observer: RiskFactorObserver,
    ) -> ContractState:
        """STF_PRF: Principal Redemption Amount Fixing - update Prnxt from array.

        This event updates the Prnxt state variable based on the array schedule.
        We need to find which array index corresponds to this time.
        """
        # Find the appropriate Prnxt value for this time
        if attrs.array_pr_anchor and attrs.array_pr_next:
            # Find the anchor index for this time
            prnxt_value = attrs.next_principal_redemption_amount or 0.0

            # Find which array segment we're in
            for i, anchor in enumerate(attrs.array_pr_anchor):
                if time >= anchor:
                    prnxt_value = attrs.array_pr_next[i]

            new_prnxt = jnp.array(prnxt_value, dtype=jnp.float32)
        else:
            new_prnxt = state.prnxt

        return state.replace(sd=time, prnxt=new_prnxt)

    def _stf_rr(
        self,
        state: ContractState,
        attrs: ContractAttributes,
        time: ActusDateTime,
        risk_factor_observer: RiskFactorObserver,
    ) -> ContractState:
        """STF_RR: Rate Reset - update interest rate from observer or array."""
        # Try to get rate from array schedule first
        if attrs.array_rr_anchor and attrs.array_rate:
            # Find which array segment we're in
            rate = attrs.nominal_interest_rate or 0.0
            for i, anchor in enumerate(attrs.array_rr_anchor):
                if time >= anchor:
                    rate = attrs.array_rate[i]
            new_rate = jnp.array(rate, dtype=jnp.float32)
        else:
            # Get from observer
            risk_factor = risk_factor_observer.observe(time)
            new_rate = jnp.array(risk_factor, dtype=jnp.float32)

        return state.replace(sd=time, ipnr=new_rate)

    def _stf_rrf(
        self,
        state: ContractState,
        attrs: ContractAttributes,
        time: ActusDateTime,
        risk_factor_observer: RiskFactorObserver,
    ) -> ContractState:
        """STF_RRF: Rate Reset Fixing - fix interest rate from array."""
        # Similar to RR but only from array
        if attrs.array_rr_anchor and attrs.array_rate:
            rate = attrs.nominal_interest_rate or 0.0
            for i, anchor in enumerate(attrs.array_rr_anchor):
                if time >= anchor:
                    rate = attrs.array_rate[i]
            new_rate = jnp.array(rate, dtype=jnp.float32)
        else:
            new_rate = state.ipnr

        return state.replace(sd=time, ipnr=new_rate)

    def _stf_sc(
        self,
        state: ContractState,
        attrs: ContractAttributes,
        time: ActusDateTime,
        risk_factor_observer: RiskFactorObserver,
    ) -> ContractState:
        """STF_SC: Scaling - not yet implemented."""
        return state.replace(sd=time)

    def _stf_ce(
        self,
        state: ContractState,
        attrs: ContractAttributes,
        time: ActusDateTime,
        risk_factor_observer: RiskFactorObserver,
    ) -> ContractState:
        """STF_CE: Credit Event - not yet implemented."""
        return state.replace(sd=time)


class ExoticLinearAmortizerContract(BaseContract):
    """LAX (Exotic Linear Amortizer) contract implementation.

    LAX is the most complex amortizing contract, supporting flexible array schedules
    for principal redemption, interest payments, and rate resets.

    ACTUS Reference:
        ACTUS v1.1 Section 7.3 - LAX: Exotic Linear Amortizer
    """

    def __init__(
        self,
        attributes: ContractAttributes,
        risk_factor_observer: RiskFactorObserver,
        child_contract_observer: Any | None = None,
    ):
        """Initialize LAX contract.

        Args:
            attributes: Contract attributes
            risk_factor_observer: Risk factor observer for rate updates
            child_contract_observer: Optional child contract observer

        Raises:
            ValueError: If contract_type is not LAX
        """
        if attributes.contract_type != ContractType.LAX:
            raise ValueError(f"Contract type must be LAX, got {attributes.contract_type.value}")

        super().__init__(
            attributes=attributes,
            risk_factor_observer=risk_factor_observer,
            child_contract_observer=child_contract_observer,
        )

    def initialize_state(self) -> ContractState:
        """Initialize LAX contract state.

        Initializes all state variables including Prnxt and Ipcb (same as LAM).

        Returns:
            Initial contract state
        """
        role_sign = contract_role_sign(self.attributes.contract_role)

        # Initialize Prnxt (next principal redemption amount)
        prnxt = jnp.array(
            role_sign * (self.attributes.next_principal_redemption_amount or 0.0),
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

    def get_payoff_function(self, event_type: Any) -> LAXPayoffFunction:
        """Get LAX payoff function.

        Args:
            event_type: Type of event (not used, all events use same POF)

        Returns:
            LAX payoff function instance
        """
        return LAXPayoffFunction(
            contract_role=self.attributes.contract_role,
            currency=self.attributes.currency,
            settlement_currency=None,
        )

    def get_state_transition_function(self, event_type: Any) -> LAXStateTransitionFunction:
        """Get LAX state transition function.

        Args:
            event_type: Type of event (not used, all events use same STF)

        Returns:
            LAX state transition function instance
        """
        return LAXStateTransitionFunction()

    def generate_event_schedule(self) -> EventSchedule:
        """Generate complete event schedule for LAX contract.

        LAX uses array schedules to generate PR, PI, PRF, IP, RR, and RRF events.

        Returns:
            EventSchedule with all contract events
        """
        events = []
        attributes = self.attributes
        ied = attributes.initial_exchange_date
        md = attributes.maturity_date

        if not ied or not md:
            return EventSchedule(events=[], contract_id=attributes.contract_id)

        # AD: Analysis Date
        events.append(
            ContractEvent(
                event_type=EventType.AD,
                event_time=attributes.status_date,
                payoff=jnp.array(0.0, dtype=jnp.float32),
                currency=attributes.currency or "XXX",
            )
        )

        # IED: Initial Exchange Date
        events.append(
            ContractEvent(
                event_type=EventType.IED,
                event_time=ied,
                payoff=jnp.array(0.0, dtype=jnp.float32),
                currency=attributes.currency or "XXX",
            )
        )

        # PR/PI Schedule: Generated from array schedules with ARINCDEC filter
        if (
            attributes.array_pr_anchor
            and attributes.array_pr_cycle
            and attributes.array_increase_decrease
        ):
            # PRF: Anchor dates where Prnxt is fixed
            prf_events = attributes.array_pr_anchor
            for time in prf_events:
                if ied < time <= md:
                    events.append(
                        ContractEvent(
                            event_type=EventType.PRF,
                            event_time=time,
                            payoff=jnp.array(0.0, dtype=jnp.float32),
                            currency=attributes.currency or "XXX",
                        )
                    )

            # PR: Principal Redemption (ARINCDEC='DEC')
            pr_schedule = generate_array_schedule(
                anchors=attributes.array_pr_anchor,
                cycles=attributes.array_pr_cycle,
                end=md,
                filter_values=attributes.array_increase_decrease,
                filter_target="DEC",
            )
            for time in pr_schedule:
                if ied < time <= md:
                    events.append(
                        ContractEvent(
                            event_type=EventType.PR,
                            event_time=time,
                            payoff=jnp.array(0.0, dtype=jnp.float32),
                            currency=attributes.currency or "XXX",
                        )
                    )

            # PI: Principal Increase (ARINCDEC='INC')
            pi_schedule = generate_array_schedule(
                anchors=attributes.array_pr_anchor,
                cycles=attributes.array_pr_cycle,
                end=md,
                filter_values=attributes.array_increase_decrease,
                filter_target="INC",
            )
            for time in pi_schedule:
                if ied < time <= md:
                    events.append(
                        ContractEvent(
                            event_type=EventType.PI,
                            event_time=time,
                            payoff=jnp.array(0.0, dtype=jnp.float32),
                            currency=attributes.currency or "XXX",
                        )
                    )

        # IP Schedule: Generated from array schedules
        if attributes.array_ip_anchor and attributes.array_ip_cycle:
            ip_schedule = generate_array_schedule(
                anchors=attributes.array_ip_anchor,
                cycles=attributes.array_ip_cycle,
                end=md,
            )
            for time in ip_schedule:
                if ied < time <= md:
                    events.append(
                        ContractEvent(
                            event_type=EventType.IP,
                            event_time=time,
                            payoff=jnp.array(0.0, dtype=jnp.float32),
                            currency=attributes.currency or "XXX",
                        )
                    )

        # RR/RRF Schedule: Generated from array schedules with ARFIXVAR filter
        if attributes.array_rr_anchor and attributes.array_rr_cycle:
            if attributes.array_fixed_variable:
                # RR: Rate Reset (ARFIXVAR='V')
                rr_schedule = generate_array_schedule(
                    anchors=attributes.array_rr_anchor,
                    cycles=attributes.array_rr_cycle,
                    end=md,
                    filter_values=attributes.array_fixed_variable,
                    filter_target="V",
                )
                for time in rr_schedule:
                    if ied < time <= md:
                        events.append(
                            ContractEvent(
                                event_type=EventType.RR,
                                event_time=time,
                                payoff=jnp.array(0.0, dtype=jnp.float32),
                                currency=attributes.currency or "XXX",
                            )
                        )

                # RRF: Rate Reset Fixing (ARFIXVAR='F')
                rrf_schedule = generate_array_schedule(
                    anchors=attributes.array_rr_anchor,
                    cycles=attributes.array_rr_cycle,
                    end=md,
                    filter_values=attributes.array_fixed_variable,
                    filter_target="F",
                )
                for time in rrf_schedule:
                    if ied < time <= md:
                        events.append(
                            ContractEvent(
                                event_type=EventType.RRF,
                                event_time=time,
                                payoff=jnp.array(0.0, dtype=jnp.float32),
                                currency=attributes.currency or "XXX",
                            )
                        )
            else:
                # No filter - default to RR
                rr_schedule = generate_array_schedule(
                    anchors=attributes.array_rr_anchor,
                    cycles=attributes.array_rr_cycle,
                    end=md,
                )
                for time in rr_schedule:
                    if ied < time <= md:
                        events.append(
                            ContractEvent(
                                event_type=EventType.RR,
                                event_time=time,
                                payoff=jnp.array(0.0, dtype=jnp.float32),
                                currency=attributes.currency or "XXX",
                            )
                        )

        # IPCB Schedule: Only for IPCB='NTL' mode
        if attributes.interest_calculation_base == "NTL":
            if (
                attributes.interest_calculation_base_cycle
                and attributes.interest_calculation_base_anchor
            ):
                ipcb_schedule = generate_schedule(
                    start=attributes.interest_calculation_base_anchor,
                    cycle=attributes.interest_calculation_base_cycle,
                    end=md,
                )
                for time in ipcb_schedule:
                    if ied < time <= md:
                        events.append(
                            ContractEvent(
                                event_type=EventType.IPCB,
                                event_time=time,
                                payoff=jnp.array(0.0, dtype=jnp.float32),
                                currency=attributes.currency or "XXX",
                            )
                        )

        # MD: Maturity Date
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
