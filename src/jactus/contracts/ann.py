"""Annuity (ANN) contract implementation.

This module implements the ANN contract type - an amortizing loan with constant
total payments (principal + interest). ANN extends NAM with automatic calculation
of payment amounts using the ACTUS annuity formula.

ACTUS Reference:
    ACTUS v1.1 Section 7.5 - ANN: Annuity

Key Features:
    - Constant total payment amount (principal + interest stays constant)
    - Automatic calculation of Prnxt using ACTUS annuity formula
    - Payment recalculation after rate resets (RR, RRF events)
    - Same negative amortization handling as NAM
    - Most common amortizing contract (standard mortgages)

Annuity Formula:
    The ACTUS annuity formula calculates the constant payment amount:
        A(s, T, n, a, r) = (n + a) / Σ[∏((1 + Y_i × r)^-1)]

    Where:
        s = start time
        T = maturity
        n = notional principal
        a = accrued interest
        r = nominal interest rate
        Y_i = year fraction for period i

Example:
    >>> from jactus.contracts import create_contract
    >>> from jactus.core import ContractAttributes, ContractType, ContractRole
    >>> from jactus.core import ActusDateTime, DayCountConvention
    >>> from jactus.observers import ConstantRiskFactorObserver
    >>>
    >>> attrs = ContractAttributes(
    ...     contract_id="MORTGAGE-001",
    ...     contract_type=ContractType.ANN,
    ...     contract_role=ContractRole.RPA,
    ...     status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
    ...     initial_exchange_date=ActusDateTime(2024, 1, 15, 0, 0, 0),
    ...     maturity_date=ActusDateTime(2054, 1, 15, 0, 0, 0),  # 30 years
    ...     currency="USD",
    ...     notional_principal=300000.0,
    ...     nominal_interest_rate=0.065,
    ...     day_count_convention=DayCountConvention.A360,
    ...     principal_redemption_cycle="1M",  # Monthly payments
    ...     interest_calculation_base="NT"
    ... )
    >>>
    >>> rf_obs = ConstantRiskFactorObserver(constant_value=0.065)
    >>> contract = create_contract(attrs, rf_obs)
    >>> result = contract.simulate()
    >>> # Payment amount automatically calculated to fully amortize loan
"""

from typing import Any

import jax.numpy as jnp

from jactus.contracts.base import BaseContract
from jactus.contracts.nam import NAMPayoffFunction, NAMStateTransitionFunction
from jactus.core import (
    ActusDateTime,
    ContractAttributes,
    ContractEvent,
    ContractState,
    ContractType,
    EventSchedule,
    EventType,
)
from jactus.functions import BaseStateTransitionFunction
from jactus.observers import RiskFactorObserver
from jactus.utilities import (
    calculate_actus_annuity,
    contract_role_sign,
    generate_schedule,
)

# ANN uses the same payoff function as NAM
ANNPayoffFunction = NAMPayoffFunction


class ANNStateTransitionFunction(BaseStateTransitionFunction):
    """State transition function for ANN contracts.

    Extends NAM state transitions with payment recalculation after rate changes.
    The key difference from NAM is that RR and RRF events recalculate Prnxt using
    the annuity formula to maintain constant total payments.

    ACTUS Reference:
        ACTUS v1.1 Section 7.5 - ANN State Transition Functions
    """

    def __init__(self):
        """Initialize ANN state transition function."""
        super().__init__()
        # Delegate to NAM for most state transitions
        self._nam_stf = NAMStateTransitionFunction()

    def transition_state(
        self,
        event_type: Any,
        state: ContractState,
        attributes: ContractAttributes,
        time: ActusDateTime,
        risk_factor_observer: RiskFactorObserver,
        child_contract_observer: Any | None = None,
    ) -> ContractState:
        """Apply state transition for ANN event.

        Args:
            event_type: Type of event
            state: Contract state before event
            attributes: Contract attributes
            time: Event time
            risk_factor_observer: Observer for market data

        Returns:
            New contract state after event
        """
        # For RR and RRF, we need to recalculate Prnxt
        if event_type == EventType.RR:
            return self._stf_rr(state, attributes, time, risk_factor_observer)
        if event_type == EventType.RRF:
            return self._stf_rrf(state, attributes, time, risk_factor_observer)

        # All other events use NAM state transitions
        return self._nam_stf.transition_state(
            event_type, state, attributes, time, risk_factor_observer
        )

    def _stf_rr(
        self,
        state: ContractState,
        attrs: ContractAttributes,
        time: ActusDateTime,
        risk_factor_observer: RiskFactorObserver,
        child_contract_observer: Any | None = None,
    ) -> ContractState:
        """STF_RR: Rate Reset - update interest rate and recalculate payment.

        Formula:
            Ipnr = observed_rate (from risk factor observer)
            Ipac = Ipac + Y(Sd, t) × Ipnr_old × Ipcb
            Prnxt = recalculated using annuity formula with new rate

        Key Feature: Recalculates Prnxt to maintain constant total payment.
        """
        # First apply NAM's RR state transition (accrues interest, updates rate)
        new_state = self._nam_stf._stf_rr(state, attrs, time, risk_factor_observer)

        # Then recalculate Prnxt using annuity formula with new rate
        # Need to get remaining PR schedule from current time to maturity
        if attrs.principal_redemption_cycle and attrs.maturity_date:
            pr_schedule = generate_schedule(
                start=time,
                cycle=attrs.principal_redemption_cycle,
                end=attrs.maturity_date,
            )
            # Filter to future events only
            pr_schedule = [d for d in pr_schedule if d > time]

            if pr_schedule:
                # Calculate new payment amount
                new_prnxt = calculate_actus_annuity(
                    start=time,
                    pr_schedule=pr_schedule,
                    notional=float(new_state.nt),
                    accrued_interest=float(new_state.ipac),
                    rate=float(new_state.ipnr),
                    day_count_convention=attrs.day_count_convention or attrs.day_count_convention,
                )

                # Update prnxt in state
                role_sign = contract_role_sign(attrs.contract_role)
                new_state = new_state.replace(
                    prnxt=jnp.array(role_sign * new_prnxt, dtype=jnp.float32)
                )

        return new_state

    def _stf_rrf(
        self,
        state: ContractState,
        attrs: ContractAttributes,
        time: ActusDateTime,
        risk_factor_observer: RiskFactorObserver,
        child_contract_observer: Any | None = None,
    ) -> ContractState:
        """STF_RRF: Rate Reset Fixing - fix interest rate and recalculate payment.

        Formula:
            Ipnr = fixed_rate (from attributes)
            Ipac = Ipac + Y(Sd, t) × Ipnr_old × Ipcb
            Prnxt = recalculated using annuity formula with fixed rate

        Key Feature: Recalculates Prnxt to maintain constant total payment.
        """
        # First apply NAM's RRF state transition (accrues interest, fixes rate)
        new_state = self._nam_stf._stf_rrf(state, attrs, time, risk_factor_observer)

        # Then recalculate Prnxt using annuity formula with fixed rate
        if attrs.principal_redemption_cycle and attrs.maturity_date:
            pr_schedule = generate_schedule(
                start=time,
                cycle=attrs.principal_redemption_cycle,
                end=attrs.maturity_date,
            )
            # Filter to future events only
            pr_schedule = [d for d in pr_schedule if d > time]

            if pr_schedule:
                # Calculate new payment amount
                new_prnxt = calculate_actus_annuity(
                    start=time,
                    pr_schedule=pr_schedule,
                    notional=float(new_state.nt),
                    accrued_interest=float(new_state.ipac),
                    rate=float(new_state.ipnr),
                    day_count_convention=attrs.day_count_convention or attrs.day_count_convention,
                )

                # Update prnxt in state
                role_sign = contract_role_sign(attrs.contract_role)
                new_state = new_state.replace(
                    prnxt=jnp.array(role_sign * new_prnxt, dtype=jnp.float32)
                )

        return new_state


class AnnuityContract(BaseContract):
    """Annuity (ANN) contract.

    Amortizing loan with constant total payment amount (principal + interest).
    The payment amount is automatically calculated using the ACTUS annuity formula
    to ensure the loan is fully amortized by maturity.

    ACTUS Reference:
        ACTUS v1.1 Section 7.5 - ANN: Annuity

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
        """Initialize ANN contract.

        Args:
            attributes: Contract attributes
            risk_factor_observer: Risk factor observer

        Raises:
            ValueError: If contract_type is not ANN
            ValueError: If required attributes missing
        """
        if attributes.contract_type != ContractType.ANN:
            raise ValueError(f"Contract type must be ANN, got {attributes.contract_type}")

        # Validate required attributes
        if not attributes.initial_exchange_date:
            raise ValueError("initial_exchange_date required for ANN")
        if not attributes.principal_redemption_cycle:
            raise ValueError("principal_redemption_cycle required for ANN")
        if not attributes.maturity_date:
            raise ValueError("maturity_date required for ANN")

        # ANN can auto-calculate Prnxt if not provided
        # (will be calculated in initialize_state)

        super().__init__(
            attributes=attributes,
            risk_factor_observer=risk_factor_observer,
            child_contract_observer=child_contract_observer,
        )

    def generate_event_schedule(self) -> EventSchedule:
        """Generate ANN event schedule.

        ANN schedule is same as NAM.

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

        # 3. Interest Payment schedule (ends one period before PR schedule for NAM/ANN)
        if self.attributes.interest_payment_cycle:
            ip_schedule = generate_schedule(
                start=self.attributes.interest_payment_anchor
                or self.attributes.initial_exchange_date,
                cycle=self.attributes.interest_payment_cycle,
                end=self.attributes.maturity_date,
            )

            # For NAM/ANN: IP schedule ends one period before last PR event
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
        """Initialize ANN contract state.

        Key Feature: If Prnxt is not provided in attributes, calculate it using
        the ACTUS annuity formula to ensure constant total payments.

        Returns:
            Initial contract state
        """
        role_sign = contract_role_sign(self.attributes.contract_role)

        # Calculate Prnxt if not provided
        if self.attributes.next_principal_redemption_amount is not None:
            # Use provided value
            prnxt = role_sign * self.attributes.next_principal_redemption_amount
        else:
            # Calculate using annuity formula
            if (
                self.attributes.principal_redemption_cycle
                and self.attributes.maturity_date
                and self.attributes.initial_exchange_date
            ):
                pr_schedule = generate_schedule(
                    start=self.attributes.principal_redemption_anchor
                    or self.attributes.initial_exchange_date,
                    cycle=self.attributes.principal_redemption_cycle,
                    end=self.attributes.maturity_date,
                )
                # Filter to future PR events (after IED)
                pr_schedule = [
                    d
                    for d in pr_schedule
                    if d > self.attributes.initial_exchange_date
                    and d < self.attributes.maturity_date
                ]

                if pr_schedule:
                    prnxt_amount = calculate_actus_annuity(
                        start=self.attributes.initial_exchange_date,
                        pr_schedule=pr_schedule,
                        notional=self.attributes.notional_principal,
                        accrued_interest=0.0,  # No accrued interest at inception
                        rate=self.attributes.nominal_interest_rate or 0.0,
                        day_count_convention=self.attributes.day_count_convention
                        or self.attributes.day_count_convention,
                    )
                    prnxt = role_sign * prnxt_amount
                else:
                    prnxt = 0.0
            else:
                prnxt = 0.0

        return ContractState(
            sd=self.attributes.status_date,
            tmd=self.attributes.maturity_date,
            nt=jnp.array(0.0, dtype=jnp.float32),  # Set at IED
            ipnr=jnp.array(0.0, dtype=jnp.float32),  # Set at IED
            ipac=jnp.array(0.0, dtype=jnp.float32),
            feac=jnp.array(0.0, dtype=jnp.float32),
            nsc=jnp.array(1.0, dtype=jnp.float32),
            isc=jnp.array(1.0, dtype=jnp.float32),
            prnxt=jnp.array(prnxt, dtype=jnp.float32),
            ipcb=jnp.array(0.0, dtype=jnp.float32),  # Set at IED
        )

    def get_payoff_function(self, event_type: Any) -> ANNPayoffFunction:
        """Get ANN payoff function.

        ANN uses the same payoff function as NAM.

        Args:
            event_type: Type of event (not used, all events use same POF)

        Returns:
            ANN payoff function instance (same as NAM)
        """
        return ANNPayoffFunction(
            contract_role=self.attributes.contract_role,
            currency=self.attributes.currency,
            settlement_currency=None,
        )

    def get_state_transition_function(self, event_type: Any) -> ANNStateTransitionFunction:
        """Get ANN state transition function.

        Args:
            event_type: Type of event (not used, all events use same STF)

        Returns:
            ANN state transition function instance
        """
        return ANNStateTransitionFunction()
