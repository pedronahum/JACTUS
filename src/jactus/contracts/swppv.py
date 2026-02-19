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

from jactus.contracts.base import BaseContract
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
from jactus.functions import BasePayoffFunction, BaseStateTransitionFunction
from jactus.observers import RiskFactorObserver
from jactus.utilities.conventions import DayCountConvention, year_fraction
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
        if event_type == EventType.PR:
            return self._pof_pr(state, attributes, time, risk_factor_observer)
        if event_type == EventType.IP:
            return self._pof_ip(state, attributes, time, risk_factor_observer)
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
        role_sign = 1.0 if attributes.contract_role in (ContractRole.RPA, ContractRole.RFL) else -1.0

        return jnp.array(role_sign * net_accrual, dtype=jnp.float32)

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

        Early termination may have a mark-to-market settlement.
        For now, zero payoff (would need market value calculation).
        """
        return jnp.array(0.0, dtype=jnp.float32)

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
            payoff=0.0,
            currency=attributes.currency or "USD",
        )

        if event_type == EventType.AD:
            return self._stf_ad(state_pre, event, attributes, risk_factor_observer)
        if event_type == EventType.IED:
            return self._stf_ied(state_pre, event, attributes, risk_factor_observer)
        if event_type == EventType.PR:
            return self._stf_pr(state_pre, event, attributes, risk_factor_observer)
        if event_type == EventType.IP:
            return self._stf_ip(state_pre, event, attributes, risk_factor_observer)
        if event_type == EventType.RR:
            return self._stf_rr(state_pre, event, attributes, risk_factor_observer)
        if event_type == EventType.TD:
            return self._stf_td(state_pre, event, attributes, risk_factor_observer)
        if event_type == EventType.CE:
            return self._stf_ce(state_pre, event, attributes, risk_factor_observer)

        # Unknown event, return state unchanged
        return state_pre

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
                ipac = net accrual (ipac1 - ipac2)
        """
        dcc = attributes.day_count_convention or DayCountConvention.A360
        yf = year_fraction(state.sd, time, dcc)
        nt = float(state.nt)

        # Fixed leg accrual: uses IPNR (nominal_interest_rate)
        fixed_rate = attributes.nominal_interest_rate or 0.0
        ipac1 = float(state.ipac1) if state.ipac1 is not None else 0.0
        new_ipac1 = ipac1 + yf * fixed_rate * nt

        # Floating leg accrual: uses current ipnr (updated by rate resets)
        ipac2 = float(state.ipac2) if state.ipac2 is not None else 0.0
        new_ipac2 = ipac2 + yf * float(state.ipnr) * nt

        # Net accrual (fixed - floating)
        new_ipac = new_ipac1 - new_ipac2

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
        new_ipac1, new_ipac2, new_ipac = self._accrue_legs(
            state, event.event_time, attributes
        )

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
        new_ipac1, new_ipac2, new_ipac = self._accrue_legs(
            state, event.event_time, attributes
        )

        # Get rate reset parameters
        rrmlt = attributes.rate_reset_multiplier if attributes.rate_reset_multiplier is not None else 1.0
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

        Accrue up to termination, then zero out.
        """
        new_ipac1, new_ipac2, new_ipac = self._accrue_legs(
            state, event.event_time, attributes
        )

        return ContractState(
            tmd=event.event_time,
            sd=event.event_time,
            nt=jnp.array(0.0, dtype=jnp.float32),
            ipnr=jnp.array(0.0, dtype=jnp.float32),
            ipac=jnp.array(new_ipac, dtype=jnp.float32),
            feac=state.feac,
            nsc=state.nsc,
            isc=state.isc,
            ipac1=jnp.array(new_ipac1, dtype=jnp.float32),
            ipac2=jnp.array(new_ipac2, dtype=jnp.float32),
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

        if attributes.rate_reset_cycle is None:
            raise ValueError("rate_reset_cycle (RRCL) is required for floating leg")

        if attributes.maturity_date is None:
            raise ValueError("maturity_date (MD) is required")

        super().__init__(attributes, risk_factor_observer, child_contract_observer)

    def generate_event_schedule(self) -> EventSchedule:
        """Generate event schedule for SWPPV contract.

        Returns:
            EventSchedule with all contract events
        """
        events = []

        # IED: Initial Exchange Date (no notional exchange)
        if self.attributes.initial_exchange_date:
            events.append(
                ContractEvent(
                    event_type=EventType.IED,
                    event_time=self.attributes.initial_exchange_date,
                    payoff=0.0,
                    currency=self.attributes.currency or "USD",
                )
            )

        # Generate Rate Reset schedule
        if self.attributes.rate_reset_cycle and self.attributes.rate_reset_anchor:
            rr_times = generate_schedule(
                start=self.attributes.rate_reset_anchor,
                end=self.attributes.maturity_date,
                cycle=self.attributes.rate_reset_cycle,
            )
            for rr_time in rr_times:
                events.append(
                    ContractEvent(
                        event_type=EventType.RR,
                        event_time=rr_time,
                        payoff=0.0,
                        currency=self.attributes.currency or "USD",
                    )
                )

        # Generate Interest Payment schedule (net settlement only)
        # Use IED as anchor if no specific anchor is set
        ip_anchor = (
            self.attributes.interest_calculation_base_anchor
            or self.attributes.initial_exchange_date
        )
        if self.attributes.interest_payment_cycle and ip_anchor:
            ip_times = generate_schedule(
                start=ip_anchor,
                end=self.attributes.maturity_date,
                cycle=self.attributes.interest_payment_cycle,
            )

            for ip_time in ip_times:
                events.append(
                    ContractEvent(
                        event_type=EventType.IP,
                        event_time=ip_time,
                        payoff=0.0,
                        currency=self.attributes.currency or "USD",
                    )
                )

        # MD: Maturity Date
        if self.attributes.maturity_date:
            events.append(
                ContractEvent(
                    event_type=EventType.MD,
                    event_time=self.attributes.maturity_date,
                    payoff=0.0,
                    currency=self.attributes.currency or "USD",
                )
            )

        # Analysis dates
        if self.attributes.analysis_dates:
            for ad_time in self.attributes.analysis_dates:
                events.append(
                    ContractEvent(
                        event_type=EventType.AD,
                        event_time=ad_time,
                        payoff=0.0,
                        currency=self.attributes.currency or "USD",
                    )
                )

        # Termination date
        if self.attributes.termination_date:
            events.append(
                ContractEvent(
                    event_type=EventType.TD,
                    event_time=self.attributes.termination_date,
                    payoff=0.0,
                    currency=self.attributes.currency or "USD",
                )
            )

        # Sort events by time
        events.sort(key=lambda e: (e.event_time.year, e.event_time.month, e.event_time.day))

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
