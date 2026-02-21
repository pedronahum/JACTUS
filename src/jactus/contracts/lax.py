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
from jactus.utilities import contract_role_sign, generate_schedule, year_fraction


def generate_array_schedule(
    anchors: list[ActusDateTime],
    cycles: list[str] | None,
    end: ActusDateTime,
    filter_values: list[str] | None = None,
    filter_target: str | None = None,
) -> list[ActusDateTime]:
    """Generate schedule from array of anchors and cycles.

    If cycles is None or empty, each anchor date is treated as a single point event.
    If cycles is provided, each (anchor, cycle) pair generates a recurring sub-schedule
    bounded by the next anchor's start date (segment boundaries).

    Args:
        anchors: Array of anchor dates (start dates for each sub-schedule)
        cycles: Array of cycles (one per anchor), or None for point events
        end: End date (maturity date)
        filter_values: Optional array of filter values (e.g., ARINCDEC)
        filter_target: Optional target value to filter for (e.g., "DEC")

    Returns:
        Union of all sub-schedules, sorted and deduplicated
    """
    if not anchors:
        return []

    # Point events: no cycles, just return anchors (after filtering)
    if not cycles:
        all_events = []
        for i, anchor in enumerate(anchors):
            if filter_values is not None and filter_target is not None:
                if filter_values[i] != filter_target:
                    continue
            all_events.append(anchor)
        all_events = sorted(set(all_events))
        return [d for d in all_events if d <= end]

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

        # Determine segment end: next anchor in the FULL array (not just filtered ones)
        # or overall end if this is the last segment
        segment_end = end
        if i + 1 < len(anchors):
            segment_end = anchors[i + 1]

        # Generate sub-schedule bounded by segment end
        sub_schedule = generate_schedule(start=anchor, cycle=cycle, end=segment_end)

        # Filter to events at or after anchor but before segment_end
        # (segment_end is the start of the next segment, so exclude it)
        sub_schedule = [d for d in sub_schedule if anchor <= d < segment_end]

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
        nt = attrs.notional_principal or 0.0
        pdied = attrs.premium_discount_at_ied or 0.0
        return jnp.array(role_sign * (-1) * (nt + pdied), dtype=jnp.float32)

    def _pof_pr(
        self, state: ContractState, attrs: ContractAttributes, time: ActusDateTime
    ) -> jnp.ndarray:
        """POF_PR: Principal Redemption - pay fixed principal amount.

        No role_sign — state.prnxt is already signed.
        """
        return state.nsc * state.prnxt

    def _pof_pi(
        self, state: ContractState, attrs: ContractAttributes, time: ActusDateTime
    ) -> jnp.ndarray:
        """POF_PI: Principal Increase - receive additional principal (negative PR).

        PI payoff is the negative of PR — the sign is already handled
        by the event type. No role_sign — state.prnxt is already signed.
        """
        return -state.nsc * state.prnxt

    def _pof_md(
        self, state: ContractState, attrs: ContractAttributes, time: ActusDateTime
    ) -> jnp.ndarray:
        """POF_MD: Maturity - pay remaining principal.

        Interest is paid by the IP event at maturity, not MD.
        No role_sign — state.nt is already signed.
        """
        return state.nsc * state.nt

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
        return state.feac

    def _pof_prd(
        self, state: ContractState, attrs: ContractAttributes, time: ActusDateTime
    ) -> jnp.ndarray:
        """POF_PRD: Purchase - not yet implemented."""
        return jnp.array(0.0, dtype=jnp.float32)

    def _pof_td(
        self, state: ContractState, attrs: ContractAttributes, time: ActusDateTime
    ) -> jnp.ndarray:
        """POF_TD: Termination - pay notional and accrued interest.

        No role_sign — state vars are already signed.
        """
        yf = year_fraction(state.sd, time, attrs.day_count_convention or attrs.day_count_convention)
        ipcb = state.ipcb if state.ipcb is not None else state.nt
        accrued = yf * state.ipnr * ipcb
        return state.nsc * (state.nt + state.ipac + accrued)

    def _pof_ip(
        self, state: ContractState, attrs: ContractAttributes, time: ActusDateTime
    ) -> jnp.ndarray:
        """POF_IP: Interest Payment - pay accrued interest on IPCB.

        No role_sign — state vars are already signed.
        """
        yf = year_fraction(state.sd, time, attrs.day_count_convention or attrs.day_count_convention)
        ipcb = state.ipcb if state.ipcb is not None else state.nt
        accrued = yf * state.ipnr * ipcb
        return state.isc * (state.ipac + accrued)

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

        # Initialize prnxt from array or single value (signed by role)
        prnxt_val = attrs.next_principal_redemption_amount
        if prnxt_val is None and attrs.array_pr_next:
            prnxt_val = attrs.array_pr_next[0]
        prnxt = jnp.array(role_sign * (prnxt_val or 0.0), dtype=jnp.float32)

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

        Same as LAM: Nt -= Prnxt (both are signed state variables).
        """
        yf = year_fraction(state.sd, time, attrs.day_count_convention or attrs.day_count_convention)

        # Calculate accrued interest using current IPCB
        ipcb = state.ipcb if state.ipcb is not None else state.nt
        new_ipac = state.ipac + yf * state.ipnr * ipcb

        # Reduce notional by prnxt (both signed, cap at remaining notional)
        effective_prnxt = jnp.sign(state.prnxt) * jnp.minimum(
            jnp.abs(state.prnxt), jnp.abs(state.nt))
        new_nt = state.nt - effective_prnxt

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

        Opposite of PR: Nt += Prnxt (both are signed state variables).
        """
        yf = year_fraction(state.sd, time, attrs.day_count_convention or attrs.day_count_convention)

        # Calculate accrued interest using current IPCB
        ipcb = state.ipcb if state.ipcb is not None else state.nt
        new_ipac = state.ipac + yf * state.ipnr * ipcb

        # Increase notional by prnxt (both signed)
        new_nt = state.nt + state.prnxt

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
        Prnxt is a signed state variable (role_sign applied).
        """
        role_sign = contract_role_sign(attrs.contract_role)
        if attrs.array_pr_anchor and attrs.array_pr_next:
            prnxt_value = attrs.next_principal_redemption_amount or 0.0

            # Find which array segment we're in
            for i, anchor in enumerate(attrs.array_pr_anchor):
                if time >= anchor:
                    prnxt_value = attrs.array_pr_next[i]

            new_prnxt = jnp.array(role_sign * prnxt_value, dtype=jnp.float32)
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
        """STF_RR: Rate Reset - accrue interest, then update rate.

        For LAX with array_rate, the array value acts as the spread for each segment.
        """
        yf = year_fraction(state.sd, time, attrs.day_count_convention or attrs.day_count_convention)
        ipcb = state.ipcb if state.ipcb is not None else state.nt
        new_ipac = state.ipac + yf * state.ipnr * ipcb

        # Get new rate from market observation
        identifier = attrs.rate_reset_market_object or "RATE"
        observed = risk_factor_observer.observe_risk_factor(identifier, time, state, attrs)
        multiplier = attrs.rate_reset_multiplier if attrs.rate_reset_multiplier is not None else 1.0

        # Use array_rate as spread if available, otherwise use rate_reset_spread
        spread = attrs.rate_reset_spread if attrs.rate_reset_spread is not None else 0.0
        if attrs.array_rate and attrs.array_rr_anchor:
            for i, anchor in enumerate(attrs.array_rr_anchor):
                if time >= anchor and i < len(attrs.array_rate):
                    spread = attrs.array_rate[i]
        new_rate = multiplier * observed + spread

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
        """STF_RRF: Rate Reset Fixing - fix interest rate from array.

        Accrue interest, then set rate from array schedule.
        """
        yf = year_fraction(state.sd, time, attrs.day_count_convention or attrs.day_count_convention)
        ipcb = state.ipcb if state.ipcb is not None else state.nt
        new_ipac = state.ipac + yf * state.ipnr * ipcb

        if attrs.array_rate:
            rate = attrs.nominal_interest_rate or 0.0
            if attrs.array_rr_anchor:
                for i, anchor in enumerate(attrs.array_rr_anchor):
                    if time >= anchor and i < len(attrs.array_rate):
                        rate = attrs.array_rate[i]
            new_rate = jnp.array(rate, dtype=jnp.float32)
        else:
            new_rate = state.ipnr

        return state.replace(sd=time, ipac=new_ipac, ipnr=new_rate)

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

        # Initialize Prnxt from single value or first array value
        prnxt_val = self.attributes.next_principal_redemption_amount
        if prnxt_val is None and self.attributes.array_pr_next:
            prnxt_val = self.attributes.array_pr_next[0]
        prnxt = jnp.array(role_sign * (prnxt_val or 0.0), dtype=jnp.float32)

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
        pr_cycles = attributes.array_pr_cycle  # May be None for point events
        if attributes.array_pr_anchor and attributes.array_increase_decrease:
            # Note: No PRF events generated — the simulate() override injects
            # prnxt from the array before each PR/PI event automatically.

            # PR: Principal Redemption (ARINCDEC='DEC')
            pr_schedule = generate_array_schedule(
                anchors=attributes.array_pr_anchor,
                cycles=pr_cycles,
                end=md,
                filter_values=attributes.array_increase_decrease,
                filter_target="DEC",
            )
            for time in pr_schedule:
                if ied < time < md:
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
                cycles=pr_cycles,
                end=md,
                filter_values=attributes.array_increase_decrease,
                filter_target="INC",
            )
            for time in pi_schedule:
                if ied < time < md:
                    events.append(
                        ContractEvent(
                            event_type=EventType.PI,
                            event_time=time,
                            payoff=jnp.array(0.0, dtype=jnp.float32),
                            currency=attributes.currency or "XXX",
                        )
                    )

        # IP Schedule: Generated from array schedules (cycles optional for point events)
        if attributes.array_ip_anchor:
            ip_schedule = generate_array_schedule(
                anchors=attributes.array_ip_anchor,
                cycles=attributes.array_ip_cycle,  # May be None
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
        # Cycles are optional (point events at anchor dates if no cycles)
        if attributes.array_rr_anchor:
            rr_cycles = attributes.array_rr_cycle  # May be None
            if attributes.array_fixed_variable:
                # RR: Rate Reset (ARFIXVAR='V')
                rr_schedule = generate_array_schedule(
                    anchors=attributes.array_rr_anchor,
                    cycles=rr_cycles,
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
                    cycles=rr_cycles,
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
                    cycles=rr_cycles,
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

        # IP at maturity if not already in schedule
        ip_times = {e.event_time for e in events if e.event_type == EventType.IP}
        if md not in ip_times:
            events.append(
                ContractEvent(
                    event_type=EventType.IP,
                    event_time=md,
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

        # Sort events by time, then by ACTUS processing order within same time
        # ACTUS order for LAX: PRF→IPCB→PR/PI→IPCI→IP→FP→RR/RRF→SC→MD
        # (RR/RRF after IP so rate change takes effect in next period)
        event_order = {
            EventType.AD: 0,
            EventType.IED: 1,
            EventType.PRF: 2,
            EventType.IPCB: 3,
            EventType.PR: 4,
            EventType.PI: 4,
            EventType.IPCI: 5,
            EventType.IP: 6,
            EventType.FP: 7,
            EventType.PP: 8,
            EventType.PY: 8,
            EventType.RR: 9,
            EventType.RRF: 9,
            EventType.SC: 10,
            EventType.TD: 11,
            EventType.MD: 12,
        }
        events.sort(key=lambda e: (e.event_time, event_order.get(e.event_type, 99)))

        return EventSchedule(events=events, contract_id=attributes.contract_id)

    def _get_prnxt_for_time(self, time: ActusDateTime) -> float | None:
        """Look up the prnxt value from array for a given event time.

        Returns the prnxt from the most recent array segment anchor at or before time.
        Returns None if no array is defined.
        """
        attrs = self.attributes
        if not attrs.array_pr_anchor or not attrs.array_pr_next:
            return None
        # Find the most recent anchor <= time
        prnxt_val = attrs.array_pr_next[0]
        for i, anchor in enumerate(attrs.array_pr_anchor):
            if time >= anchor and i < len(attrs.array_pr_next):
                prnxt_val = attrs.array_pr_next[i]
        return prnxt_val

    def simulate(
        self,
        risk_factor_observer: RiskFactorObserver | None = None,
        child_contract_observer: Any | None = None,
    ) -> SimulationHistory:
        """Simulate LAX contract with array-aware prnxt injection.

        Before each PR/PI event, updates state.prnxt from the array schedule
        so the correct principal amount is used without explicit PRF events.
        """
        from jactus.observers import ChildContractObserver

        risk_obs = risk_factor_observer or self.risk_factor_observer
        role_sign = contract_role_sign(self.attributes.contract_role)

        state = self.initialize_state()
        initial_state = state
        events_with_states = []

        schedule = self.get_events()

        for event in schedule.events:
            stf = self.get_state_transition_function(event.event_type)
            pof = self.get_payoff_function(event.event_type)
            calc_time = event.calculation_time or event.event_time

            # Inject prnxt from array before PR/PI events
            if event.event_type in (EventType.PR, EventType.PI, EventType.PRF):
                prnxt_val = self._get_prnxt_for_time(calc_time)
                if prnxt_val is not None:
                    state = state.replace(
                        prnxt=jnp.array(role_sign * prnxt_val, dtype=jnp.float32)
                    )

            payoff = pof(
                event_type=event.event_type,
                state=state,
                attributes=self.attributes,
                time=calc_time,
                risk_factor_observer=risk_obs,
            )

            state_post = stf(
                event_type=event.event_type,
                state_pre=state,
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
