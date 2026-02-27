"""Plain Vanilla Interest Rate Swap (SWPPV) contract implementation.

This module implements a plain vanilla interest rate swap where one party pays a
fixed rate and receives a floating rate (or vice versa). The swap exchanges interest
payments on a notional amount without exchanging the principal.

Key Features:
    - Fixed leg: Uses nominal_interest_rate (IPNR)
    - Floating leg: Uses nominal_interest_rate_2 (IPNR2) with rate resets
    - Separate accrual tracking (ipac1 for fixed, ipac2 for floating)
    - Net or gross settlement modes
    - No notional exchange at inception or maturity

Example:
    >>> from jactus.contracts import PlainVanillaSwapContract
    >>> from jactus.core import ContractAttributes, ActusDateTime
    >>> from jactus.observers import ConstantRiskFactorObserver
    >>>
    >>> # Receive fixed, pay floating
    >>> attrs = ContractAttributes(
    ...     contract_id="SWAP-001",
    ...     contract_type=ContractType.SWPPV,
    ...     contract_role=ContractRole.RPA,  # Receive fixed
    ...     status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
    ...     contract_deal_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
    ...     initial_exchange_date=ActusDateTime(2024, 1, 15, 0, 0, 0),
    ...     maturity_date=ActusDateTime(2029, 1, 15, 0, 0, 0),
    ...     notional_principal=1000000.0,
    ...     nominal_interest_rate=0.05,  # Fixed leg: 5%
    ...     nominal_interest_rate_2=0.03,  # Floating leg initial: 3%
    ...     interest_payment_cycle="P6M",  # Semi-annual
    ...     rate_reset_cycle="P3M",  # Quarterly resets
    ...     delivery_settlement="D",  # Net settlement
    ... )
    >>> rf_obs = ConstantRiskFactorObserver(0.04)
    >>> swap = PlainVanillaSwapContract(attrs, rf_obs)
    >>> cashflows = swap.simulate(rf_obs)
"""

from typing import Any

import jax.numpy as jnp

from jactus.contracts.base import BaseContract, SimulationHistory
from jactus.core import (
    ActusDateTime,
    ContractAttributes,
    ContractEvent,
    ContractPerformance,
    ContractRole,
    ContractState,
    ContractType,
    EventSchedule,
    EventType,
)
from jactus.core.types import DayCountConvention
from jactus.functions import BasePayoffFunction, BaseStateTransitionFunction
from jactus.observers import RiskFactorObserver
from jactus.observers.behavioral import BehaviorRiskFactorObserver
from jactus.observers.scenario import Scenario
from jactus.utilities import year_fraction
from jactus.utilities.schedules import generate_schedule


class PlainVanillaSwapPayoffFunction(BasePayoffFunction):
    """Payoff function for SWPPV contracts.

    Calculates cashflows for fixed and floating leg interest payments.
    """

    def calculate_payoff(
        self,
        event_type: EventType,
        state: ContractState,
        attributes: ContractAttributes,
        time: ActusDateTime,
        risk_factor_observer: RiskFactorObserver,
    ) -> jnp.ndarray:
        """Dispatcher for payoff functions."""
        if event_type == EventType.AD:
            return self._pof_ad(state, attributes, time, risk_factor_observer)
        if event_type == EventType.IED:
            return self._pof_ied(state, attributes, time, risk_factor_observer)
        if event_type == EventType.PRD:
            return self._pof_prd(state, attributes, time, risk_factor_observer)
        if event_type == EventType.PR:
            return self._pof_pr(state, attributes, time, risk_factor_observer)
        if event_type == EventType.IP:
            return self._pof_ip(state, attributes, time, risk_factor_observer)
        if event_type == EventType.IPFX:
            return self._pof_ipfx(state, attributes, time, risk_factor_observer)
        if event_type == EventType.IPFL:
            return self._pof_ipfl(state, attributes, time, risk_factor_observer)
        if event_type == EventType.MD:
            return self._pof_md(state, attributes, time, risk_factor_observer)
        if event_type == EventType.RR:
            return self._pof_rr(state, attributes, time, risk_factor_observer)
        if event_type == EventType.TD:
            return self._pof_td(state, attributes, time, risk_factor_observer)
        if event_type == EventType.CE:
            return self._pof_ce(state, attributes, time, risk_factor_observer)
        # Unknown event type
        return jnp.array(0.0, dtype=jnp.float32)

    def _pof_ad(
        self,
        state: ContractState,
        attributes: ContractAttributes,
        time: ActusDateTime,
        risk_factor_observer: RiskFactorObserver,
    ) -> jnp.ndarray:
        """POF_AD_SWPPV: Analysis Date payoff.

        Analysis dates have zero payoff.
        """
        return jnp.array(0.0, dtype=jnp.float32)

    def _pof_ied(
        self,
        state: ContractState,
        attributes: ContractAttributes,
        time: ActusDateTime,
        risk_factor_observer: RiskFactorObserver,
    ) -> jnp.ndarray:
        """POF_IED_SWPPV: Initial Exchange Date payoff.

        No notional exchange for plain vanilla swaps.
        """
        return jnp.array(0.0, dtype=jnp.float32)

    def _pof_prd(
        self,
        state: ContractState,
        attributes: ContractAttributes,
        time: ActusDateTime,
        risk_factor_observer: RiskFactorObserver,
    ) -> jnp.ndarray:
        """POF_PRD_SWPPV: Purchase Date payoff.

        Formula: POF_PRD = R(CNTRL) × (-PPRD)
        """
        from jactus.utilities import contract_role_sign

        pprd = attributes.price_at_purchase_date or 0.0
        role_sign = contract_role_sign(attributes.contract_role)
        return jnp.array(role_sign * (-pprd), dtype=jnp.float32)

    def _pof_md(
        self,
        state: ContractState,
        attributes: ContractAttributes,
        time: ActusDateTime,
        risk_factor_observer: RiskFactorObserver,
    ) -> jnp.ndarray:
        """POF_MD_SWPPV: Maturity Date payoff.

        No notional exchange at maturity for plain vanilla swaps.
        """
        return jnp.array(0.0, dtype=jnp.float32)

    def _pof_pr(
        self,
        state: ContractState,
        attributes: ContractAttributes,
        time: ActusDateTime,
        risk_factor_observer: RiskFactorObserver,
    ) -> jnp.ndarray:
        """POF_PR_SWPPV: Principal Redemption payoff.

        No principal redemption for swaps (no notional exchange).
        """
        return jnp.array(0.0, dtype=jnp.float32)

    def _pof_ip(
        self,
        state: ContractState,
        attributes: ContractAttributes,
        time: ActusDateTime,
        risk_factor_observer: RiskFactorObserver,
    ) -> jnp.ndarray:
        """POF_IP_SWPPV: Interest Payment (Net settlement).

        Computes the net payment including accrual up to payment time.

        POF is called with pre-event state, so we accrue from state.sd to time,
        then compute: net = (ipac1 + fixed_accrual) - (ipac2 + floating_accrual)

        For RPA (receive fixed, pay floating):
            payoff = net  (positive when fixed > floating)
        For RPL (pay fixed, receive floating):
            payoff = -net
        """
        dcc = attributes.day_count_convention or DayCountConvention.A360
        yf = year_fraction(state.sd, time, dcc)
        nt = float(state.nt)

        # Accrue fixed leg up to payment time
        fixed_rate = attributes.nominal_interest_rate or 0.0
        ipac1 = float(state.ipac1) if state.ipac1 is not None else 0.0
        total_ipac1 = ipac1 + yf * fixed_rate * nt

        # Accrue floating leg up to payment time
        floating_rate = float(state.ipnr)
        ipac2 = float(state.ipac2) if state.ipac2 is not None else 0.0
        total_ipac2 = ipac2 + yf * floating_rate * nt

        # Net accrual (fixed - floating)
        net_accrual = total_ipac1 - total_ipac2

        # Contract role sign: RPA receives fixed leg, RPL pays it
        role_sign = (
            1.0 if attributes.contract_role in (ContractRole.RPA, ContractRole.RFL) else -1.0
        )

        return jnp.array(role_sign * net_accrual, dtype=jnp.float32)

    def _pof_ipfx(
        self,
        state: ContractState,
        attributes: ContractAttributes,
        time: ActusDateTime,
        risk_factor_observer: RiskFactorObserver,
    ) -> jnp.ndarray:
        """POF_IPFX_SWPPV: Fixed Leg Interest Payment.

        Formula:
            POF_IPFX = R(CNTRL) × (Ipac1 + Y(Sd, t) × IPNR × Nt)

        Where:
            Ipac1: Accumulated fixed leg accrual
            IPNR: Fixed nominal interest rate
            R(CNTRL): Role sign (PFL=-1 pays fixed, RFL=+1 receives fixed)
        """
        from jactus.utilities import contract_role_sign

        dcc = attributes.day_count_convention or DayCountConvention.A360
        yf = year_fraction(state.sd, time, dcc)
        nt = float(state.nt)

        fixed_rate = attributes.nominal_interest_rate or 0.0
        ipac1 = float(state.ipac1) if state.ipac1 is not None else 0.0
        total_fixed = ipac1 + yf * fixed_rate * nt

        role_sign = contract_role_sign(attributes.contract_role)

        return jnp.array(role_sign * total_fixed, dtype=jnp.float32)

    def _pof_ipfl(
        self,
        state: ContractState,
        attributes: ContractAttributes,
        time: ActusDateTime,
        risk_factor_observer: RiskFactorObserver,
    ) -> jnp.ndarray:
        """POF_IPFL_SWPPV: Floating Leg Interest Payment.

        Formula:
            POF_IPFL = -R(CNTRL) × (Ipac2 + Y(Sd, t) × Ipnr × Nt)

        Where:
            Ipac2: Accumulated floating leg accrual
            Ipnr: Current floating rate (from state, updated by RR)
            R(CNTRL): Role sign (PFL=-1, so -(-1) = +1 receives floating)
        """
        from jactus.utilities import contract_role_sign

        dcc = attributes.day_count_convention or DayCountConvention.A360
        yf = year_fraction(state.sd, time, dcc)
        nt = float(state.nt)

        floating_rate = float(state.ipnr)
        ipac2 = float(state.ipac2) if state.ipac2 is not None else 0.0
        total_floating = ipac2 + yf * floating_rate * nt

        role_sign = contract_role_sign(attributes.contract_role)

        return jnp.array(-role_sign * total_floating, dtype=jnp.float32)

    def _pof_rr(
        self,
        state: ContractState,
        attributes: ContractAttributes,
        time: ActusDateTime,
        risk_factor_observer: RiskFactorObserver,
    ) -> jnp.ndarray:
        """POF_RR_SWPPV: Rate Reset payoff.

        Rate resets have zero payoff (only update state).
        """
        return jnp.array(0.0, dtype=jnp.float32)

    def _pof_td(
        self,
        state: ContractState,
        attributes: ContractAttributes,
        time: ActusDateTime,
        risk_factor_observer: RiskFactorObserver,
    ) -> jnp.ndarray:
        """POF_TD_SWPPV: Termination Date payoff.

        For SWPPV, PTD is the mark-to-market settlement amount
        (already directional).
        """
        ptd = attributes.price_at_termination_date or 0.0
        return jnp.array(ptd, dtype=jnp.float32)

    def _pof_ce(
        self,
        state: ContractState,
        attributes: ContractAttributes,
        time: ActusDateTime,
        risk_factor_observer: RiskFactorObserver,
    ) -> jnp.ndarray:
        """POF_CE_SWPPV: Credit Event payoff.

        Credit events have zero payoff (would trigger termination).
        """
        return jnp.array(0.0, dtype=jnp.float32)


class PlainVanillaSwapStateTransitionFunction(BaseStateTransitionFunction):
    """State transition function for SWPPV contracts.

    Manages accrual tracking for fixed and floating legs.
    """

    def transition_state(
        self,
        event_type: EventType,
        state_pre: ContractState,
        attributes: ContractAttributes,
        time: ActusDateTime,
        risk_factor_observer: RiskFactorObserver,
    ) -> ContractState:
        """Calculate state transition for swap events."""
        # Create event for helper functions
        event = ContractEvent(
            event_type=event_type,
            event_time=time,
            payoff=jnp.array(0.0, dtype=jnp.float32),
            currency=attributes.currency or "USD",
        )

        if event_type == EventType.AD:
            return self._stf_ad(state_pre, event, attributes, risk_factor_observer)
        if event_type == EventType.IED:
            return self._stf_ied(state_pre, event, attributes, risk_factor_observer)
        if event_type == EventType.PRD:
            return self._stf_prd(state_pre, event, attributes, risk_factor_observer)
        if event_type == EventType.PR:
            return self._stf_pr(state_pre, event, attributes, risk_factor_observer)
        if event_type == EventType.IP:
            return self._stf_ip(state_pre, event, attributes, risk_factor_observer)
        if event_type == EventType.IPFX:
            return self._stf_ipfx(state_pre, event, attributes, risk_factor_observer)
        if event_type == EventType.IPFL:
            return self._stf_ipfl(state_pre, event, attributes, risk_factor_observer)
        if event_type == EventType.MD:
            return self._stf_md(state_pre, event, attributes, risk_factor_observer)
        if event_type == EventType.RR:
            return self._stf_rr(state_pre, event, attributes, risk_factor_observer)
        if event_type == EventType.TD:
            return self._stf_td(state_pre, event, attributes, risk_factor_observer)
        if event_type == EventType.CE:
            return self._stf_ce(state_pre, event, attributes, risk_factor_observer)

        # Unknown event, return state unchanged
        return state_pre

    @staticmethod
    def _adjust_eod_time(time: ActusDateTime) -> ActusDateTime:
        """Adjust end-of-day times (23:59:59) to next day midnight for accrual."""
        if time.hour == 23 and time.minute == 59:
            from datetime import timedelta

            py_dt = time.to_datetime() + timedelta(seconds=1)
            return ActusDateTime(py_dt.year, py_dt.month, py_dt.day, 0, 0, 0)
        return time

    def _accrue_legs(
        self,
        state: ContractState,
        time: ActusDateTime,
        attributes: ContractAttributes,
    ) -> tuple[float, float, float]:
        """Accrue interest for both legs up to the given time.

        Returns:
            Tuple of (new_ipac1, new_ipac2, new_ipac) where:
                ipac1 = fixed leg accrual
                ipac2 = floating leg accrual
                ipac = R(CNTRL) × ipac1 (signed fixed leg accrual)
        """
        from jactus.utilities import contract_role_sign

        dcc = attributes.day_count_convention or DayCountConvention.A360
        # Adjust end-of-day times for correct day count
        adj_time = self._adjust_eod_time(time)
        yf = year_fraction(state.sd, adj_time, dcc)
        nt = float(state.nt)

        # Fixed leg accrual: uses IPNR (nominal_interest_rate)
        fixed_rate = attributes.nominal_interest_rate or 0.0
        ipac1 = float(state.ipac1) if state.ipac1 is not None else 0.0
        new_ipac1 = ipac1 + yf * fixed_rate * nt

        # Floating leg accrual: uses current ipnr (updated by rate resets)
        ipac2 = float(state.ipac2) if state.ipac2 is not None else 0.0
        new_ipac2 = ipac2 + yf * float(state.ipnr) * nt

        # ipac = R(CNTRL) × ipac1 (signed fixed leg accrual for ACTUS state)
        role_sign = contract_role_sign(attributes.contract_role)
        new_ipac = role_sign * new_ipac1

        return new_ipac1, new_ipac2, new_ipac

    def _stf_ad(
        self,
        state: ContractState,
        event: ContractEvent,
        attributes: ContractAttributes,
        risk_factor_observer: RiskFactorObserver,
    ) -> ContractState:
        """STF_AD_SWPPV: Analysis Date state transition.

        Accrue interest for both fixed and floating legs.
        """
        new_ipac1, new_ipac2, new_ipac = self._accrue_legs(state, event.event_time, attributes)

        return ContractState(
            tmd=state.tmd,
            sd=event.event_time,
            nt=state.nt,
            ipnr=state.ipnr,
            ipac=jnp.array(new_ipac, dtype=jnp.float32),
            feac=state.feac,
            nsc=state.nsc,
            isc=state.isc,
            ipac1=jnp.array(new_ipac1, dtype=jnp.float32),
            ipac2=jnp.array(new_ipac2, dtype=jnp.float32),
            prf=state.prf if hasattr(state, "prf") else ContractPerformance.PF,
        )

    def _stf_ied(
        self,
        state: ContractState,
        event: ContractEvent,
        attributes: ContractAttributes,
        risk_factor_observer: RiskFactorObserver,
    ) -> ContractState:
        """STF_IED_SWPPV: Initial Exchange Date state transition.

        Initialize accruals to zero.
        """
        return ContractState(
            tmd=attributes.maturity_date or event.event_time,
            sd=event.event_time,
            nt=jnp.array(attributes.notional_principal or 1.0, dtype=jnp.float32),
            ipnr=jnp.array(
                attributes.nominal_interest_rate_2 or 0.0, dtype=jnp.float32
            ),  # Floating rate
            ipac=jnp.array(0.0, dtype=jnp.float32),
            feac=jnp.array(0.0, dtype=jnp.float32),
            nsc=jnp.array(1.0, dtype=jnp.float32),
            isc=jnp.array(1.0, dtype=jnp.float32),
            ipac1=jnp.array(0.0, dtype=jnp.float32),  # Fixed leg accrual
            ipac2=jnp.array(0.0, dtype=jnp.float32),  # Floating leg accrual
            prf=attributes.contract_performance or ContractPerformance.PF,
        )

    def _stf_prd(
        self,
        state: ContractState,
        event: ContractEvent,
        attributes: ContractAttributes,
        risk_factor_observer: RiskFactorObserver,
    ) -> ContractState:
        """STF_PRD_SWPPV: Purchase Date state transition.

        Accrue interest from status date to purchase date.
        """
        new_ipac1, new_ipac2, new_ipac = self._accrue_legs(state, event.event_time, attributes)
        return ContractState(
            tmd=state.tmd,
            sd=event.event_time,
            nt=state.nt,
            ipnr=state.ipnr,
            ipac=jnp.array(new_ipac, dtype=jnp.float32),
            feac=state.feac,
            nsc=state.nsc,
            isc=state.isc,
            ipac1=jnp.array(new_ipac1, dtype=jnp.float32),
            ipac2=jnp.array(new_ipac2, dtype=jnp.float32),
            prf=state.prf if hasattr(state, "prf") else ContractPerformance.PF,
        )

    def _stf_md(
        self,
        state: ContractState,
        event: ContractEvent,
        attributes: ContractAttributes,
        risk_factor_observer: RiskFactorObserver,
    ) -> ContractState:
        """STF_MD_SWPPV: Maturity Date state transition.

        Reset notional to zero.
        """
        return ContractState(
            tmd=event.event_time,
            sd=event.event_time,
            nt=jnp.array(0.0, dtype=jnp.float32),
            ipnr=state.ipnr,
            ipac=jnp.array(0.0, dtype=jnp.float32),
            feac=state.feac,
            nsc=state.nsc,
            isc=state.isc,
            ipac1=jnp.array(0.0, dtype=jnp.float32),
            ipac2=jnp.array(0.0, dtype=jnp.float32),
            prf=state.prf if hasattr(state, "prf") else ContractPerformance.PF,
        )

    def _stf_pr(
        self,
        state: ContractState,
        event: ContractEvent,
        attributes: ContractAttributes,
        risk_factor_observer: RiskFactorObserver,
    ) -> ContractState:
        """STF_PR_SWPPV: Principal Redemption state transition.

        No state change for swaps (no principal).
        """
        return state

    def _stf_ip(
        self,
        state: ContractState,
        event: ContractEvent,
        attributes: ContractAttributes,
        risk_factor_observer: RiskFactorObserver,
    ) -> ContractState:
        """STF_IP_SWPPV: Interest Payment state transition.

        Accrue both legs up to payment time, then reset all accruals to zero.
        The payoff function reads ipac (net) before this transition resets it.
        """
        # First accrue up to payment time (payoff function will read state before transition)
        # After payment, reset all accruals to zero
        return ContractState(
            tmd=state.tmd,
            sd=event.event_time,
            nt=state.nt,
            ipnr=state.ipnr,
            ipac=jnp.array(0.0, dtype=jnp.float32),
            feac=state.feac,
            nsc=state.nsc,
            isc=state.isc,
            ipac1=jnp.array(0.0, dtype=jnp.float32),
            ipac2=jnp.array(0.0, dtype=jnp.float32),
            prf=state.prf if hasattr(state, "prf") else ContractPerformance.PF,
        )

    def _stf_ipfx(
        self,
        state: ContractState,
        event: ContractEvent,
        attributes: ContractAttributes,
        risk_factor_observer: RiskFactorObserver,
    ) -> ContractState:
        """STF_IPFX_SWPPV: Fixed Leg Interest Payment state transition.

        Accrue fixed leg up to payment time, then reset fixed leg accrual.
        """
        dcc = attributes.day_count_convention or DayCountConvention.A360
        yf = year_fraction(state.sd, event.event_time, dcc)
        nt = float(state.nt)

        # Accrue floating leg (don't reset it - IPFL handles that)
        floating_rate = float(state.ipnr)
        ipac2 = float(state.ipac2) if state.ipac2 is not None else 0.0
        new_ipac2 = ipac2 + yf * floating_rate * nt

        return ContractState(
            tmd=state.tmd,
            sd=event.event_time,
            nt=state.nt,
            ipnr=state.ipnr,
            ipac=jnp.array(0.0, dtype=jnp.float32),
            feac=state.feac,
            nsc=state.nsc,
            isc=state.isc,
            ipac1=jnp.array(0.0, dtype=jnp.float32),  # Reset fixed accrual
            ipac2=jnp.array(new_ipac2, dtype=jnp.float32),
            prf=state.prf if hasattr(state, "prf") else ContractPerformance.PF,
        )

    def _stf_ipfl(
        self,
        state: ContractState,
        event: ContractEvent,
        attributes: ContractAttributes,
        risk_factor_observer: RiskFactorObserver,
    ) -> ContractState:
        """STF_IPFL_SWPPV: Floating Leg Interest Payment state transition.

        Reset floating leg accrual after payment.
        Note: IPFL follows IPFX on the same date, so fixed accrual is already reset.
        """
        return ContractState(
            tmd=state.tmd,
            sd=event.event_time,
            nt=state.nt,
            ipnr=state.ipnr,
            ipac=jnp.array(0.0, dtype=jnp.float32),
            feac=state.feac,
            nsc=state.nsc,
            isc=state.isc,
            ipac1=state.ipac1,
            ipac2=jnp.array(0.0, dtype=jnp.float32),  # Reset floating accrual
            prf=state.prf if hasattr(state, "prf") else ContractPerformance.PF,
        )

    def _stf_rr(
        self,
        state: ContractState,
        event: ContractEvent,
        attributes: ContractAttributes,
        risk_factor_observer: RiskFactorObserver,
    ) -> ContractState:
        """STF_RR_SWPPV: Rate Reset state transition.

        Accrue both legs up to reset time, then update floating rate.

        Formula:
            Ipac1_t = Ipac1_(t-) + Y(Sd_(t-), t) * IPNR * Nt
            Ipac2_t = Ipac2_(t-) + Y(Sd_(t-), t) * Ipnr_(t-) * Nt
            Ipnr_t = min(max(RRMLT * O_rf(RRMO, t) + RRSP, RRLF), RRLC)
        """
        # First accrue both legs up to reset time using old rates
        new_ipac1, new_ipac2, new_ipac = self._accrue_legs(state, event.event_time, attributes)

        # Get rate reset parameters
        rrmlt = (
            attributes.rate_reset_multiplier
            if attributes.rate_reset_multiplier is not None
            else 1.0
        )
        rrsp = attributes.rate_reset_spread if attributes.rate_reset_spread is not None else 0.0
        rrmo = attributes.rate_reset_market_object or ""

        # Observe market rate
        market_rate = risk_factor_observer.observe_risk_factor(rrmo, event.event_time)

        # Calculate new floating rate
        new_rate = rrmlt * float(market_rate) + rrsp

        # Apply caps and floors if specified
        if attributes.rate_reset_floor is not None:
            new_rate = max(new_rate, attributes.rate_reset_floor)
        if attributes.rate_reset_cap is not None:
            new_rate = min(new_rate, attributes.rate_reset_cap)

        return ContractState(
            tmd=state.tmd,
            sd=event.event_time,
            nt=state.nt,
            ipnr=jnp.array(new_rate, dtype=jnp.float32),
            ipac=jnp.array(new_ipac, dtype=jnp.float32),
            feac=state.feac,
            nsc=state.nsc,
            isc=state.isc,
            ipac1=jnp.array(new_ipac1, dtype=jnp.float32),
            ipac2=jnp.array(new_ipac2, dtype=jnp.float32),
            prf=state.prf if hasattr(state, "prf") else ContractPerformance.PF,
        )

    def _stf_td(
        self,
        state: ContractState,
        event: ContractEvent,
        attributes: ContractAttributes,
        risk_factor_observer: RiskFactorObserver,
    ) -> ContractState:
        """STF_TD_SWPPV: Termination Date state transition.

        Zero notional and accruals, preserve rate.
        """
        return ContractState(
            tmd=event.event_time,
            sd=event.event_time,
            nt=jnp.array(0.0, dtype=jnp.float32),
            ipnr=state.ipnr,  # Preserve current floating rate
            ipac=jnp.array(0.0, dtype=jnp.float32),
            feac=state.feac,
            nsc=state.nsc,
            isc=state.isc,
            ipac1=jnp.array(0.0, dtype=jnp.float32),
            ipac2=jnp.array(0.0, dtype=jnp.float32),
            prf=state.prf if hasattr(state, "prf") else ContractPerformance.PF,
        )

    def _stf_ce(
        self,
        state: ContractState,
        event: ContractEvent,
        attributes: ContractAttributes,
        risk_factor_observer: RiskFactorObserver,
    ) -> ContractState:
        """STF_CE_SWPPV: Credit Event state transition.

        Credit events don't change state (would trigger termination).
        """
        return state


class PlainVanillaSwapContract(BaseContract):
    """Plain Vanilla Interest Rate Swap (SWPPV) contract.

    Swaps fixed and floating interest rate payments on a notional amount.
    No exchange of principal occurs.

    Attributes:
        attributes: Contract terms and conditions
        risk_factor_observer: Observer for market rates
    """

    def __init__(
        self,
        attributes: ContractAttributes,
        risk_factor_observer: RiskFactorObserver,
        child_contract_observer: Any = None,
    ):
        """Initialize SWPPV contract.

        Args:
            attributes: Contract attributes
            risk_factor_observer: Observer for market data
            child_contract_observer: Not used for SWPPV

        Raises:
            ValueError: If required attributes are missing or invalid
        """
        # Validate contract type
        if attributes.contract_type != ContractType.SWPPV:
            raise ValueError(f"Expected contract_type=SWPPV, got {attributes.contract_type}")

        # Validate required attributes
        if attributes.notional_principal is None or attributes.notional_principal == 0:
            raise ValueError("notional_principal (NT) is required and must be non-zero")

        if attributes.nominal_interest_rate is None:
            raise ValueError("nominal_interest_rate (IPNR) is required for fixed leg")

        if attributes.nominal_interest_rate_2 is None:
            raise ValueError(
                "nominal_interest_rate_2 (IPNR2) is required for initial floating rate"
            )

        if attributes.interest_payment_cycle is None:
            raise ValueError("interest_payment_cycle (IPCL) is required")

        if attributes.maturity_date is None:
            raise ValueError("maturity_date (MD) is required")

        super().__init__(attributes, risk_factor_observer, child_contract_observer)

    # Event ordering for same-date events (lower = earlier)
    _EVENT_ORDER = {
        EventType.IED: 0,
        EventType.PRD: 1,
        EventType.IPFX: 2,
        EventType.IPFL: 3,
        EventType.IP: 4,
        EventType.RR: 5,
        EventType.MD: 10,
        EventType.TD: 11,
        EventType.AD: 12,
    }

    def generate_event_schedule(self) -> EventSchedule:
        """Generate event schedule for SWPPV contract.

        Returns:
            EventSchedule with all contract events
        """
        events = []
        ccy = self.attributes.currency or "USD"
        maturity = self.attributes.maturity_date

        # Determine settlement mode: D = separate (IPFX/IPFL), S = net (IP)
        ds = self.attributes.delivery_settlement or "D"
        use_separate = ds == "D"

        # IED: Initial Exchange Date
        if self.attributes.initial_exchange_date:
            events.append(
                ContractEvent(
                    event_type=EventType.IED,
                    event_time=self.attributes.initial_exchange_date,
                    payoff=jnp.array(0.0, dtype=jnp.float32),
                    currency=ccy,
                )
            )

        # Generate Rate Reset schedule (exclude maturity date)
        if self.attributes.rate_reset_cycle and self.attributes.rate_reset_anchor:
            rr_times = generate_schedule(
                start=self.attributes.rate_reset_anchor,
                end=maturity,
                cycle=self.attributes.rate_reset_cycle,
            )
            for rr_time in rr_times:
                # Exclude RR at maturity date
                if maturity and rr_time.to_iso()[:10] == maturity.to_iso()[:10]:
                    continue
                events.append(
                    ContractEvent(
                        event_type=EventType.RR,
                        event_time=rr_time,
                        payoff=jnp.array(0.0, dtype=jnp.float32),
                        currency=ccy,
                    )
                )
        elif self.attributes.rate_reset_anchor:
            # Single RR at anchor date (no cycle)
            rr_anchor = self.attributes.rate_reset_anchor
            if not maturity or rr_anchor.to_iso()[:10] != maturity.to_iso()[:10]:
                events.append(
                    ContractEvent(
                        event_type=EventType.RR,
                        event_time=rr_anchor,
                        payoff=jnp.array(0.0, dtype=jnp.float32),
                        currency=ccy,
                    )
                )

        # Generate Interest Payment schedule
        ip_anchor = (
            self.attributes.interest_payment_anchor
            or self.attributes.interest_calculation_base_anchor
            or self.attributes.initial_exchange_date
        )
        if self.attributes.interest_payment_cycle and ip_anchor:
            ip_times = generate_schedule(
                start=ip_anchor,
                end=maturity,
                cycle=self.attributes.interest_payment_cycle,
            )

            # Add maturity date as final payment if not already included
            if maturity:
                maturity_iso = maturity.to_iso()[:10]
                if not any(t.to_iso()[:10] == maturity_iso for t in ip_times):
                    ip_times.append(maturity)

            for ip_time in ip_times:
                if use_separate:
                    events.append(
                        ContractEvent(
                            event_type=EventType.IPFX,
                            event_time=ip_time,
                            payoff=jnp.array(0.0, dtype=jnp.float32),
                            currency=ccy,
                        )
                    )
                    events.append(
                        ContractEvent(
                            event_type=EventType.IPFL,
                            event_time=ip_time,
                            payoff=jnp.array(0.0, dtype=jnp.float32),
                            currency=ccy,
                        )
                    )
                else:
                    events.append(
                        ContractEvent(
                            event_type=EventType.IP,
                            event_time=ip_time,
                            payoff=jnp.array(0.0, dtype=jnp.float32),
                            currency=ccy,
                        )
                    )

        # PRD: Purchase Date
        if self.attributes.purchase_date:
            events.append(
                ContractEvent(
                    event_type=EventType.PRD,
                    event_time=self.attributes.purchase_date,
                    payoff=jnp.array(0.0, dtype=jnp.float32),
                    currency=ccy,
                )
            )

        # MD: Maturity Date
        if maturity:
            events.append(
                ContractEvent(
                    event_type=EventType.MD,
                    event_time=maturity,
                    payoff=jnp.array(0.0, dtype=jnp.float32),
                    currency=ccy,
                )
            )

        # Analysis dates
        if self.attributes.analysis_dates:
            for ad_time in self.attributes.analysis_dates:
                events.append(
                    ContractEvent(
                        event_type=EventType.AD,
                        event_time=ad_time,
                        payoff=jnp.array(0.0, dtype=jnp.float32),
                        currency=ccy,
                    )
                )

        # Termination date
        if self.attributes.termination_date:
            events.append(
                ContractEvent(
                    event_type=EventType.TD,
                    event_time=self.attributes.termination_date,
                    payoff=jnp.array(0.0, dtype=jnp.float32),
                    currency=ccy,
                )
            )

        # Sort by date, then by event type ordering (IP/IPFX/IPFL before RR)
        events.sort(
            key=lambda e: (
                e.event_time.to_iso()[:10],
                self._EVENT_ORDER.get(e.event_type, 99),
            )
        )

        return EventSchedule(
            contract_id=self.attributes.contract_id,
            events=tuple(events),
        )

    def initialize_state(self) -> ContractState:
        """Initialize contract state at status date.

        Returns:
            Initial ContractState
        """
        return ContractState(
            tmd=self.attributes.maturity_date or self.attributes.status_date,
            sd=self.attributes.status_date,
            nt=jnp.array(self.attributes.notional_principal or 1.0, dtype=jnp.float32),
            ipnr=jnp.array(self.attributes.nominal_interest_rate_2 or 0.0, dtype=jnp.float32),
            ipac=jnp.array(0.0, dtype=jnp.float32),
            feac=jnp.array(0.0, dtype=jnp.float32),
            nsc=jnp.array(1.0, dtype=jnp.float32),
            isc=jnp.array(1.0, dtype=jnp.float32),
            ipac1=jnp.array(0.0, dtype=jnp.float32),
            ipac2=jnp.array(0.0, dtype=jnp.float32),
            prf=self.attributes.contract_performance or ContractPerformance.PF,
        )

    def get_payoff_function(self, event_type: Any) -> PlainVanillaSwapPayoffFunction:
        """Get payoff function for SWPPV contract.

        Args:
            event_type: Type of event (not used, kept for interface compatibility)

        Returns:
            PlainVanillaSwapPayoffFunction instance
        """
        return PlainVanillaSwapPayoffFunction(
            contract_role=self.attributes.contract_role,
            currency=self.attributes.currency,
        )

    def get_state_transition_function(
        self, event_type: Any
    ) -> PlainVanillaSwapStateTransitionFunction:
        """Get state transition function for SWPPV contract.

        Args:
            event_type: Type of event (not used, kept for interface compatibility)

        Returns:
            PlainVanillaSwapStateTransitionFunction instance
        """
        return PlainVanillaSwapStateTransitionFunction()

    def simulate(
        self,
        risk_factor_observer: RiskFactorObserver | None = None,
        child_contract_observer: Any = None,
        scenario: Scenario | None = None,
        behavior_observers: list[BehaviorRiskFactorObserver] | None = None,
    ) -> SimulationHistory:
        """Simulate SWPPV contract.

        Overrides base to filter out events before purchaseDate and after
        terminationDate. The full event schedule is processed for state
        computation, but only visible events are returned.
        """
        result = super().simulate(
            risk_factor_observer,
            child_contract_observer,
            scenario=scenario,
            behavior_observers=behavior_observers,
        )

        # Filter events: keep only PRD onwards when purchaseDate is set
        if self.attributes.purchase_date:
            purchase_iso = self.attributes.purchase_date.to_iso()
            filtered = [e for e in result.events if e.event_time.to_iso() >= purchase_iso]
            result = SimulationHistory(
                events=filtered,
                states=result.states,
                initial_state=result.initial_state,
                final_state=result.final_state,
            )

        # Filter events: keep only up to and including TD when terminationDate is set
        if self.attributes.termination_date:
            td_iso = self.attributes.termination_date.to_iso()[:10]
            filtered = [e for e in result.events if e.event_time.to_iso()[:10] <= td_iso]
            result = SimulationHistory(
                events=filtered,
                states=result.states,
                initial_state=result.initial_state,
                final_state=result.final_state,
            )

        return result
