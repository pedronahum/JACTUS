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

import calendar as _calendar_mod
from datetime import timedelta
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
from jactus.functions import BaseStateTransitionFunction
from jactus.observers import RiskFactorObserver
from jactus.utilities import (
    calculate_actus_annuity,
    contract_role_sign,
    generate_schedule,
    year_fraction,
)

# ANN uses the same payoff function as NAM
ANNPayoffFunction = NAMPayoffFunction


def _subtract_one_cycle(dt: ActusDateTime, cycle: str) -> ActusDateTime:
    """Compute dt minus one cycle period (PRANX-)."""
    import re

    base = cycle.rstrip("+-")
    match = re.match(r"(\d+)([DMQHYW])", base)
    if not match:
        return dt
    number = int(match.group(1))
    ptype = match.group(2)

    py_dt = dt.to_datetime()
    if ptype == "D":
        result = py_dt - timedelta(days=number)
    elif ptype == "W":
        result = py_dt - timedelta(weeks=number)
    elif ptype in ("M", "Q", "H", "Y"):
        months_map = {"M": 1, "Q": 3, "H": 6, "Y": 12}
        months = number * months_map[ptype]
        new_month = py_dt.month - months
        new_year = py_dt.year
        while new_month <= 0:
            new_month += 12
            new_year -= 1
        max_day = _calendar_mod.monthrange(new_year, new_month)[1]
        new_day = min(py_dt.day, max_day)
        result = py_dt.replace(year=new_year, month=new_month, day=new_day)
    else:
        return dt

    return ActusDateTime(result.year, result.month, result.day, 0, 0, 0)


class ANNStateTransitionFunction(BaseStateTransitionFunction):
    """State transition function for ANN contracts.

    Extends NAM state transitions with payment recalculation after rate changes.
    The key difference from NAM is that RR and RRF events recalculate Prnxt using
    the annuity formula to maintain constant total payments.

    ACTUS Reference:
        ACTUS v1.1 Section 7.5 - ANN State Transition Functions
    """

    def __init__(self) -> None:
        """Initialize ANN state transition function."""
        super().__init__()
        # Delegate to NAM for most state transitions
        self._nam_stf = NAMStateTransitionFunction()

    def _build_dispatch_table(self) -> dict[EventType, Any]:
        """Build event type → handler dispatch table.

        ANN overrides RR and RRF with annuity-specific handlers that
        recalculate Prnxt. All other events delegate to NAM's dispatch.
        """
        # Start with NAM's full dispatch table
        table = self._nam_stf._build_dispatch_table()
        # Override RR and RRF with ANN-specific handlers
        table[EventType.RR] = self._stf_rr
        table[EventType.RRF] = self._stf_rrf
        table[EventType.PRF] = self._stf_prf
        return table

    def transition_state(
        self,
        event_type: Any,
        state: ContractState,
        attributes: ContractAttributes,
        time: ActusDateTime,
        risk_factor_observer: RiskFactorObserver,
        child_contract_observer: Any | None = None,
    ) -> ContractState:
        """Apply state transition for ANN event via dict dispatch.

        ANN overrides RR/RRF to recalculate Prnxt using annuity formula.
        All other events delegate to NAM's handlers.

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
            result: ContractState = handler(state, attributes, time, risk_factor_observer)
            return result
        return state

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
        new_state = self._recalculate_annuity(new_state, attrs, time)
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
        new_state = self._recalculate_annuity(new_state, attrs, time)
        return new_state

    def _stf_prf(
        self,
        state: ContractState,
        attrs: ContractAttributes,
        time: ActusDateTime,
        risk_factor_observer: RiskFactorObserver,
        child_contract_observer: Any | None = None,
    ) -> ContractState:
        """STF_PRF: Principal Redemption Amount Fixing.

        Accrue interest up to PRF date, then recalculate Prnxt using
        annuity formula with current rate and remaining schedule.

        Formula:
            Ipac = Ipac + Y(Sd, t) × Ipnr × Ipcb
            Prnxt = A(t, T, Nt, Ipac, Ipnr) (annuity formula)
        """
        # Accrue interest from last status date to PRF time
        dcc = attrs.day_count_convention or DayCountConvention.A360
        yf = year_fraction(state.sd, time, dcc)
        ipcb = state.ipcb if state.ipcb is not None else state.nt
        new_ipac = state.ipac + yf * state.ipnr * ipcb
        new_state = state.replace(sd=time, ipac=new_ipac)

        # Recalculate Prnxt using annuity formula
        new_state = self._recalculate_annuity(new_state, attrs, time)
        return new_state

    def _recalculate_annuity(
        self,
        state: ContractState,
        attrs: ContractAttributes,
        time: ActusDateTime,
    ) -> ContractState:
        """Recalculate Prnxt using ACTUS annuity formula.

        For the initial PRF (before any PR event), uses PRANX- as start with
        accrued_interest=0. For RR-triggered PRFs, uses the PRF event time as
        start and current accrued interest per the ACTUS spec:
            Prnxt = A(t, T, Nt, Ipac, Ipnr)
        """
        if not (attrs.principal_redemption_cycle and attrs.maturity_date):
            return state

        pr_anchor = attrs.principal_redemption_anchor or attrs.initial_exchange_date
        if pr_anchor is None:
            return state
        pr_cycle = attrs.principal_redemption_cycle
        # Use amortization_date for annuity calculation horizon when available
        md = attrs.amortization_date or attrs.maturity_date

        # Generate PR schedule from anchor to end
        pr_schedule = generate_schedule(start=pr_anchor, cycle=pr_cycle, end=md)
        pr_schedule = [d for d in pr_schedule if d <= md]

        # Long stub: remove last regular date before end (longer final period)
        if pr_cycle.endswith("+") and pr_schedule and md not in pr_schedule:
            pr_schedule = pr_schedule[:-1]

        # Ensure MD is included as final payment date
        if md not in pr_schedule:
            pr_schedule.append(md)

        # Determine if this is an initial PRF (no PR dates <= time)
        has_pr_before = any(d <= time for d in pr_schedule)

        if has_pr_before:
            # RR-triggered PRF: use event time as start, include accrued interest
            start = time
            ipac = abs(float(state.ipac))
        else:
            # Initial PRF: use max(IED, PRANX-1cycle) as the annuity start
            pranx_minus = _subtract_one_cycle(pr_anchor, pr_cycle)
            ied = attrs.initial_exchange_date
            start = max(ied, pranx_minus) if ied else pranx_minus
            ipac = 0.0

        # Remaining PR dates after start
        remaining = [d for d in pr_schedule if d > start]

        if remaining:
            new_prnxt = calculate_actus_annuity(
                start=start,
                pr_schedule=remaining,
                notional=abs(float(state.nt)),
                accrued_interest=ipac,
                rate=float(state.ipnr),
                day_count_convention=attrs.day_count_convention or DayCountConvention.A360,
            )

            role_sign = contract_role_sign(attrs.contract_role)
            state = state.replace(prnxt=jnp.array(role_sign * new_prnxt, dtype=jnp.float32))

        return state


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

        # Handle amortization_date: when maturity_date is absent, use amortization_date
        if not attributes.maturity_date and attributes.amortization_date:
            attributes = attributes.model_copy(
                update={"maturity_date": attributes.amortization_date}
            )

        # Derive maturity_date if not provided but PRNXT is given
        if not attributes.maturity_date and attributes.next_principal_redemption_amount:
            attributes = self._derive_maturity_date(attributes)
        elif not attributes.maturity_date:
            raise ValueError(
                "maturity_date (or amortizationDate or nextPrincipalRedemptionPayment) required for ANN"
            )

        # ANN can auto-calculate Prnxt if not provided
        # (will be calculated in initialize_state)

        super().__init__(
            attributes=attributes,
            risk_factor_observer=risk_factor_observer,
            child_contract_observer=child_contract_observer,
        )

    @staticmethod
    def _derive_maturity_date(attrs: ContractAttributes) -> ContractAttributes:
        """Derive maturity_date by simulating amortization when not provided.

        When maturityDate is not given but PRNXT (total payment) is, simulate
        the amortization: each period, interest accrues on remaining notional,
        principal = PRNXT - interest, and notional decreases. The date at which
        notional reaches 0 is the maturity date.
        """
        ied = attrs.initial_exchange_date
        assert ied is not None
        nt = attrs.notional_principal or 0.0
        prnxt = attrs.next_principal_redemption_amount or 0.0
        rate = attrs.nominal_interest_rate or 0.0
        dcc = attrs.day_count_convention or DayCountConvention.A360
        pr_anchor = attrs.principal_redemption_anchor or ied
        pr_cycle = attrs.principal_redemption_cycle
        assert pr_cycle is not None

        # Generate enough dates to cover full amortization (50 years)
        eomc = attrs.end_of_month_convention or EndOfMonthConvention.SD
        bdc = attrs.business_day_convention or BusinessDayConvention.NULL
        cal = attrs.calendar or Calendar.NO_CALENDAR
        far_future = ied.add_period("600M", EndOfMonthConvention.SD)
        pr_dates = generate_schedule(
            start=pr_anchor,
            cycle=pr_cycle,
            end=far_future,
            end_of_month_convention=eomc,
            business_day_convention=bdc,
            calendar=cal,
        )

        # Simulate amortization
        remaining = float(nt)
        prev_date = ied
        md = None
        for dt in pr_dates:
            interest = remaining * rate * year_fraction(prev_date, dt, dcc)
            principal_payment = prnxt - interest
            if principal_payment <= 0:
                # Payment doesn't cover interest - skip
                prev_date = dt
                continue
            remaining -= principal_payment
            if remaining <= 1e-6:  # Effectively zero
                md = dt
                break
            prev_date = dt

        if md is None:
            raise ValueError("Could not derive maturity_date: amortization does not converge")

        # Return new attributes with derived maturity_date
        return attrs.model_copy(update={"maturity_date": md})

    def generate_event_schedule(self) -> EventSchedule:
        """Generate ANN event schedule per ACTUS specification.

        Same as NAM/LAM schedule, with all event types supported.

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
        rr_dates = []
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

        # PRF: Principal Redemption Fixing (only when PRNXT not provided)
        if not attrs.next_principal_redemption_amount:
            pr_anchor = attrs.principal_redemption_anchor or ied
            # Initial PRF: one day before first PR date (only if PRANX > IED)
            if pr_anchor > ied:
                py_dt = pr_anchor.to_datetime() - timedelta(days=1)
                prf_initial = ActusDateTime(py_dt.year, py_dt.month, py_dt.day, 0, 0, 0)
                _add(EventType.PRF, prf_initial)

            # PRF at each RR date (to recalculate after rate change)
            if attrs.rate_reset_cycle and attrs.rate_reset_anchor:
                for dt in rr_dates:
                    if md and dt >= md:
                        break
                    _add(EventType.PRF, dt)

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
            tmd=md or attrs.status_date,
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
        """Initialize ANN contract state.

        Key Feature: If Prnxt is not provided in attributes, calculate it using
        the ACTUS annuity formula to ensure constant total payments.

        When IED < SD (contract already existed), state is initialized
        as if STF_IED already ran.

        Returns:
            Initial contract state
        """
        attrs = self.attributes
        sd = attrs.status_date
        ied = attrs.initial_exchange_date
        role_sign = contract_role_sign(attrs.contract_role)

        # Calculate Prnxt if not provided
        if attrs.next_principal_redemption_amount is not None:
            prnxt = role_sign * attrs.next_principal_redemption_amount
        else:
            # Calculate using annuity formula with PRANX- as start
            if attrs.principal_redemption_cycle and attrs.maturity_date and ied:
                pr_anchor = attrs.principal_redemption_anchor or ied
                pr_cycle = attrs.principal_redemption_cycle
                # Use amortization_date for annuity calculation horizon when provided
                annuity_end = attrs.amortization_date or attrs.maturity_date

                pr_schedule = generate_schedule(
                    start=pr_anchor,
                    cycle=pr_cycle,
                    end=annuity_end,
                )
                pr_schedule = [d for d in pr_schedule if d <= annuity_end]

                # Long stub: remove last regular date before end
                if pr_cycle.endswith("+") and pr_schedule and annuity_end not in pr_schedule:
                    pr_schedule = pr_schedule[:-1]
                if annuity_end not in pr_schedule:
                    pr_schedule.append(annuity_end)

                # Annuity start: use the later of IED or PRANX-1cycle.
                # When PRANX > IED by more than one cycle, PRANX-1cycle is
                # correct (start of the regular payment schedule). But when
                # PRANX == IED or PRANX-1cycle < IED, use IED to avoid a
                # phantom pre-loan period that distorts the annuity formula.
                pranx_minus = _subtract_one_cycle(pr_anchor, pr_cycle)
                annuity_start = max(ied, pranx_minus)

                if pr_schedule:
                    prnxt_amount = calculate_actus_annuity(
                        start=annuity_start,
                        pr_schedule=pr_schedule,
                        notional=attrs.notional_principal or 0.0,
                        accrued_interest=0.0,
                        rate=attrs.nominal_interest_rate or 0.0,
                        day_count_convention=attrs.day_count_convention or DayCountConvention.A360,
                    )
                    prnxt = role_sign * prnxt_amount
                else:
                    prnxt = 0.0
            else:
                prnxt = 0.0

        # PRD pre-simulation
        if attrs.purchase_date and ied:
            return self._pre_simulate_to_prd(attrs, jnp.array(prnxt, dtype=jnp.float32))

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
                ipac = yf * ipnr * abs(nt)
            else:
                ipac = 0.0

            ipcb_val = abs(nt)
            return ContractState(
                sd=init_sd,
                tmd=attrs.maturity_date or attrs.status_date,
                nt=jnp.array(nt, dtype=jnp.float32),
                ipnr=jnp.array(ipnr, dtype=jnp.float32),
                ipac=jnp.array(ipac, dtype=jnp.float32),
                feac=jnp.array(0.0, dtype=jnp.float32),
                nsc=jnp.array(1.0, dtype=jnp.float32),
                isc=jnp.array(1.0, dtype=jnp.float32),
                prnxt=jnp.array(prnxt, dtype=jnp.float32),
                ipcb=jnp.array(ipcb_val, dtype=jnp.float32),
            )

        return ContractState(
            sd=sd,
            tmd=attrs.maturity_date or attrs.status_date,
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
