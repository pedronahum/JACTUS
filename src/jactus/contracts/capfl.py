"""Cap-Floor (CAPFL) contract implementation.

This module implements interest rate cap and floor contracts. A cap/floor
protects against interest rate movements:
  - Cap: pays max(0, floating_rate - cap_rate) * NT * YF
  - Floor: pays max(0, floor_rate - floating_rate) * NT * YF
  - Collar: both cap and floor

The CAPFL wraps an underlier (typically PAM or SWPPV) and generates events
on the underlier's IP/RR schedule.

Example:
    >>> from jactus.contracts import CapFloorContract
    >>> from jactus.core import ContractAttributes, ActusDateTime, ContractType, ContractRole
    >>> from jactus.observers import ConstantRiskFactorObserver, MockChildContractObserver
    >>>
    >>> attrs = ContractAttributes(
    ...     contract_id="CAP-001",
    ...     contract_type=ContractType.CAPFL,
    ...     contract_role=ContractRole.BUY,
    ...     status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
    ...     maturity_date=ActusDateTime(2029, 1, 1, 0, 0, 0),
    ...     rate_reset_cap=0.06,
    ...     contract_structure='{"Underlying": "SWAP-001"}',
    ... )
    >>> rf_obs = ConstantRiskFactorObserver(0.03)
    >>> child_obs = MockChildContractObserver()
    >>> cap = CapFloorContract(attrs, rf_obs, child_obs)

References:
    ACTUS Technical Specification v1.1, Section 7.14
"""

import json
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
from jactus.observers import ChildContractObserver, RiskFactorObserver
from jactus.utilities.conventions import year_fraction
from jactus.utilities.schedules import generate_schedule


class CapFloorPayoffFunction(BasePayoffFunction):
    """Payoff function for CAPFL contracts.

    Computes cap/floor differential payoffs at IP events.
    """

    def __init__(
        self,
        contract_role: ContractRole | None = None,
        currency: str | None = None,
        settlement_currency: str | None = None,
        cap_rate: float | None = None,
        floor_rate: float | None = None,
        notional: float = 0.0,
        day_count_convention: DayCountConvention = DayCountConvention.A365,
    ):
        super().__init__(contract_role, currency, settlement_currency)
        self.cap_rate = cap_rate
        self.floor_rate = floor_rate
        self.notional = notional
        self.dcc = day_count_convention

    def calculate_payoff(
        self,
        event_type: EventType,
        state: ContractState,
        attributes: ContractAttributes,
        time: ActusDateTime,
        risk_factor_observer: RiskFactorObserver,
    ) -> jnp.ndarray:
        """Calculate cap/floor payoff.

        For IP events, computes the cap/floor differential:
          Cap:   max(0, floating_rate - cap_rate) * NT * YF
          Floor: max(0, floor_rate - floating_rate) * NT * YF
        """
        if event_type == EventType.IP:
            return self._pof_ip(state, attributes, time)
        return jnp.array(0.0, dtype=jnp.float32)

    def _pof_ip(
        self,
        state: ContractState,
        attributes: ContractAttributes,
        time: ActusDateTime,
    ) -> jnp.ndarray:
        """POF_IP_CAPFL: Interest payment - cap/floor differential."""
        floating_rate = float(state.ipnr)
        nt = self.notional
        yf = year_fraction(state.sd, time, self.dcc)

        payoff = 0.0
        if self.cap_rate is not None:
            payoff += max(0.0, floating_rate - self.cap_rate) * nt * yf
        if self.floor_rate is not None:
            payoff += max(0.0, self.floor_rate - floating_rate) * nt * yf

        # Role sign: BUY receives protection, SEL pays it
        role_sign = 1.0
        if attributes.contract_role in (ContractRole.RPL, ContractRole.ST, ContractRole.SEL):
            role_sign = -1.0

        return jnp.array(role_sign * payoff, dtype=jnp.float32)


class CapFloorStateTransitionFunction(BaseStateTransitionFunction):
    """State transition function for CAPFL contracts.

    Tracks the floating rate through RR events and advances sd at IP events.
    """

    def __init__(self, dcc: DayCountConvention = DayCountConvention.A365):
        super().__init__()
        self.dcc = dcc

    def transition_state(
        self,
        event_type: EventType,
        state_pre: ContractState,
        attributes: ContractAttributes,
        time: ActusDateTime,
        risk_factor_observer: RiskFactorObserver,
    ) -> ContractState:
        if event_type == EventType.RR:
            return self._stf_rr(state_pre, attributes, time, risk_factor_observer)
        if event_type == EventType.IP:
            return self._stf_ip(state_pre, time)
        return ContractState(
            sd=time,
            tmd=state_pre.tmd,
            nt=state_pre.nt,
            ipnr=state_pre.ipnr,
            ipac=state_pre.ipac,
            feac=state_pre.feac,
            nsc=state_pre.nsc,
            isc=state_pre.isc,
        )

    def _stf_rr(
        self,
        state_pre: ContractState,
        attributes: ContractAttributes,
        time: ActusDateTime,
        risk_factor_observer: RiskFactorObserver,
    ) -> ContractState:
        """STF_RR_CAPFL: Rate Reset - observe market rate for cap/floor tracking."""
        market_object = attributes.rate_reset_market_object or ""
        observed_rate = float(
            risk_factor_observer.observe_risk_factor(market_object, time, state_pre, attributes)
        )
        return ContractState(
            sd=time,
            tmd=state_pre.tmd,
            nt=state_pre.nt,
            ipnr=jnp.array(observed_rate, dtype=jnp.float32),
            ipac=state_pre.ipac,
            feac=state_pre.feac,
            nsc=state_pre.nsc,
            isc=state_pre.isc,
        )

    def _stf_ip(self, state_pre: ContractState, time: ActusDateTime) -> ContractState:
        """STF_IP_CAPFL: Interest Payment - advance status date."""
        return ContractState(
            sd=time,
            tmd=state_pre.tmd,
            nt=state_pre.nt,
            ipnr=state_pre.ipnr,
            ipac=jnp.array(0.0, dtype=jnp.float32),
            feac=state_pre.feac,
            nsc=state_pre.nsc,
            isc=state_pre.isc,
        )


class CapFloorContract(BaseContract):
    """Cap-Floor (CAPFL) contract.

    An interest rate cap or floor that pays the differential when
    the floating rate breaches the cap or floor rate.

    The contract either:
    1. References an underlier via contract_structure (child observer mode)
    2. Contains embedded underlier terms for standalone operation
    """

    def __init__(
        self,
        attributes: ContractAttributes,
        risk_factor_observer: RiskFactorObserver,
        child_contract_observer: ChildContractObserver | None = None,
    ):
        if attributes.contract_type != ContractType.CAPFL:
            raise ValueError(
                f"Expected contract_type=CAPFL, got {attributes.contract_type}"
            )

        if child_contract_observer is None:
            raise ValueError(
                "child_contract_observer is required for CAPFL contracts"
            )

        if attributes.contract_structure is None:
            raise ValueError(
                "contract_structure (CTST) is required and must contain Underlying"
            )

        # Parse contract structure (JSON string)
        try:
            ctst = json.loads(attributes.contract_structure)
        except (json.JSONDecodeError, TypeError) as e:
            raise ValueError(f"contract_structure must be valid JSON: {e}") from e

        if not isinstance(ctst, dict):
            raise ValueError("contract_structure must be a JSON object (dictionary)")

        if "Underlying" not in ctst:
            raise ValueError("contract_structure must contain 'Underlying' key")

        if attributes.rate_reset_cap is None and attributes.rate_reset_floor is None:
            raise ValueError(
                "At least one of rate_reset_cap (RRLC) or rate_reset_floor (RRLF) must be set"
            )

        # Parse underlier terms if embedded
        underlying = ctst["Underlying"]
        self._underlier_terms: dict[str, Any] | None = None
        if isinstance(underlying, dict):
            self._underlier_terms = underlying

        super().__init__(attributes, risk_factor_observer, child_contract_observer)

    def _parse_contract_structure(self) -> dict[str, Any]:
        return json.loads(self.attributes.contract_structure)

    def generate_event_schedule(self) -> EventSchedule:
        """Generate event schedule for CAPFL contract.

        If underlier terms are embedded, generates IP and RR schedules
        directly from those terms. Otherwise, queries the child observer.
        """
        events: list[ContractEvent] = []

        if self._underlier_terms:
            events = self._generate_standalone_schedule()
        else:
            events = self._generate_child_observer_schedule()

        # Add analysis dates
        currency = self.attributes.currency or "USD"
        if self.attributes.analysis_dates:
            for ad_time in self.attributes.analysis_dates:
                events.append(
                    ContractEvent(
                        event_type=EventType.AD,
                        event_time=ad_time,
                        payoff=jnp.array(0.0, dtype=jnp.float32),
                        currency=currency,
                    )
                )

        # Add termination date
        if self.attributes.termination_date:
            events.append(
                ContractEvent(
                    event_type=EventType.TD,
                    event_time=self.attributes.termination_date,
                    payoff=jnp.array(0.0, dtype=jnp.float32),
                    currency=currency,
                )
            )

        events.sort(
            key=lambda e: (
                e.event_time.year,
                e.event_time.month,
                e.event_time.day,
                e.sequence,
            )
        )

        return EventSchedule(
            contract_id=self.attributes.contract_id,
            events=tuple(events),
        )

    def _generate_standalone_schedule(self) -> list[ContractEvent]:
        """Generate schedule from embedded underlier terms."""
        events: list[ContractEvent] = []
        terms = self._underlier_terms
        assert terms is not None

        # Parse underlier dates/cycles
        ied_str = terms.get("initialExchangeDate")
        md_str = terms.get("maturityDate")
        if md_str is None:
            return events

        md = ActusDateTime.from_iso(md_str)
        # Use IED as schedule start, or derive from MD by stepping back
        if ied_str:
            start = ActusDateTime.from_iso(ied_str)
        else:
            # Derive schedule anchor from MD by stepping backward
            start = self._derive_start_from_md(md, terms)

        currency = self.attributes.currency or terms.get("currency", "USD")

        # IP schedule from underlier
        # For CAPFL, IP runs BEFORE RR at the same timestamp so that IP
        # uses the rate from the previous period (not the just-reset rate)
        ip_cycle_str = terms.get("cycleOfInterestPayment", "")
        if ip_cycle_str:
            cycle = self._parse_cycle(ip_cycle_str)
            ip_dates = generate_schedule(start=start, cycle=cycle, end=md)
            for ip_time in ip_dates[1:]:
                events.append(
                    ContractEvent(
                        event_type=EventType.IP,
                        event_time=ip_time,
                        payoff=jnp.array(0.0, dtype=jnp.float32),
                        currency=currency,
                        sequence=0,  # IP BEFORE RR at same time
                    )
                )

        # RR schedule from underlier
        rr_cycle_str = terms.get("cycleOfRateReset", "")
        if rr_cycle_str:
            cycle = self._parse_cycle(rr_cycle_str)
            rr_dates = generate_schedule(start=start, cycle=cycle, end=md)
            # Skip first date, include subsequent
            for rr_time in rr_dates[1:]:
                events.append(
                    ContractEvent(
                        event_type=EventType.RR,
                        event_time=rr_time,
                        payoff=jnp.array(0.0, dtype=jnp.float32),
                        currency=currency,
                        sequence=1,  # RR AFTER IP at same time
                    )
                )

        return events

    def _derive_start_from_md(self, md: ActusDateTime, terms: dict) -> ActusDateTime:
        """Derive schedule start by stepping backward from MD in cycle increments.

        When no IED is specified, find the earliest cycle-aligned date
        after the status date by stepping backward from MD.
        """
        import re

        cycle_str = terms.get("cycleOfInterestPayment", terms.get("cycleOfRateReset", ""))
        cycle = self._parse_cycle(cycle_str)

        match = re.match(r"(\d+)([DWMY])", cycle)
        if not match:
            return self.attributes.status_date

        n = int(match.group(1))
        unit = match.group(2)

        if unit == "M":
            months = n
        elif unit == "Y":
            months = n * 12
        else:
            return self.attributes.status_date

        # Step backward from MD until we pass status_date
        sd = self.attributes.status_date
        current = md
        while True:
            year = current.year
            month = current.month - months
            while month <= 0:
                year -= 1
                month += 12
            day = min(current.day, 28)
            try:
                prev = ActusDateTime(year, month, day, 0, 0, 0)
            except Exception:
                prev = ActusDateTime(year, month, 28, 0, 0, 0)
            if prev <= sd:
                return current
            current = prev

    def _generate_child_observer_schedule(self) -> list[ContractEvent]:
        """Generate schedule using child observer (legacy approach)."""
        events: list[ContractEvent] = []

        ctst = self._parse_contract_structure()
        underlier_id = ctst["Underlying"]

        uncapped_events = self.child_contract_observer.observe_events(
            underlier_id,
            self.attributes.status_date,
            None,
        )
        capped_events = self.child_contract_observer.observe_events(
            underlier_id,
            self.attributes.status_date,
            None,
        )

        uncapped_map: dict[ActusDateTime, ContractEvent] = {}
        for event in uncapped_events:
            if event.event_type == EventType.IP:
                uncapped_map[event.event_time] = event

        capped_map: dict[ActusDateTime, ContractEvent] = {}
        for event in capped_events:
            if event.event_type == EventType.IP:
                capped_map[event.event_time] = event

        all_times = set(uncapped_map.keys()) | set(capped_map.keys())
        for time in all_times:
            uncapped_payoff = (
                float(uncapped_map[time].payoff) if time in uncapped_map else 0.0
            )
            capped_payoff = (
                float(capped_map[time].payoff) if time in capped_map else 0.0
            )
            differential = abs(uncapped_payoff - capped_payoff)
            if differential > 0.0:
                events.append(
                    ContractEvent(
                        event_type=EventType.IP,
                        event_time=time,
                        payoff=jnp.array(differential, dtype=jnp.float32),
                        currency=self.attributes.currency or "USD",
                    )
                )

        if self.attributes.maturity_date:
            events.append(
                ContractEvent(
                    event_type=EventType.MD,
                    event_time=self.attributes.maturity_date,
                    payoff=jnp.array(0.0, dtype=jnp.float32),
                    currency=self.attributes.currency or "USD",
                )
            )

        return events

    @staticmethod
    def _parse_cycle(cycle_str: str) -> str:
        """Convert ACTUS ISO cycle (P3ML1) to JACTUS format (3M)."""
        s = cycle_str
        if s.startswith("P"):
            s = s[1:]
        if "L" in s:
            s = s[: s.index("L")]
        return s

    def initialize_state(self) -> ContractState:
        """Initialize CAPFL contract state."""
        # Get initial rate from underlier terms
        ipnr = 0.0
        if self._underlier_terms:
            ipnr = float(self._underlier_terms.get("nominalInterestRate", 0.0))

        return ContractState(
            tmd=self.attributes.maturity_date or self.attributes.status_date,
            sd=self.attributes.status_date,
            nt=jnp.array(1.0, dtype=jnp.float32),
            ipnr=jnp.array(ipnr, dtype=jnp.float32),
            ipac=jnp.array(0.0, dtype=jnp.float32),
            feac=jnp.array(0.0, dtype=jnp.float32),
            nsc=jnp.array(1.0, dtype=jnp.float32),
            isc=jnp.array(1.0, dtype=jnp.float32),
            prf=self.attributes.contract_performance or ContractPerformance.PF,
        )

    def get_payoff_function(self, event_type: Any) -> CapFloorPayoffFunction:
        # Get underlier's notional and DCC
        notional = 0.0
        dcc = DayCountConvention.A365
        if self._underlier_terms:
            notional = float(self._underlier_terms.get("notionalPrincipal", 0.0))
            dcc_str = self._underlier_terms.get("dayCountConvention", "A365")
            dcc = _parse_dcc(dcc_str)

        return CapFloorPayoffFunction(
            contract_role=self.attributes.contract_role,
            currency=self.attributes.currency,
            cap_rate=self.attributes.rate_reset_cap,
            floor_rate=self.attributes.rate_reset_floor,
            notional=notional,
            day_count_convention=dcc,
        )

    def get_state_transition_function(
        self, event_type: Any
    ) -> CapFloorStateTransitionFunction:
        dcc = DayCountConvention.A365
        if self._underlier_terms:
            dcc_str = self._underlier_terms.get("dayCountConvention", "A365")
            dcc = _parse_dcc(dcc_str)

        return CapFloorStateTransitionFunction(dcc=dcc)

    def simulate(
        self,
        risk_factor_observer: RiskFactorObserver | None = None,
        child_contract_observer: ChildContractObserver | None = None,
    ) -> SimulationHistory:
        """Simulate CAPFL contract.

        RR events are used internally for rate tracking but filtered from
        the output since CAPFL only exposes IP events externally.
        """
        risk_obs = risk_factor_observer or self.risk_factor_observer

        # Store market object from underlier for RR observations
        if self._underlier_terms:
            market_object = self._underlier_terms.get("marketObjectCodeOfRateReset", "")
            if market_object and not self.attributes.rate_reset_market_object:
                self.attributes.rate_reset_market_object = market_object

        result = super().simulate(risk_obs, child_contract_observer)

        # Filter out internal RR events — CAPFL only outputs IP events
        # Zero out state for IP events (CAPFL is a derivative with no own notional/rate)
        zero_state = ContractState(
            tmd=result.initial_state.tmd,
            sd=result.initial_state.sd,
            nt=jnp.array(0.0, dtype=jnp.float32),
            ipnr=jnp.array(0.0, dtype=jnp.float32),
            ipac=jnp.array(0.0, dtype=jnp.float32),
            feac=jnp.array(0.0, dtype=jnp.float32),
            nsc=jnp.array(1.0, dtype=jnp.float32),
            isc=jnp.array(1.0, dtype=jnp.float32),
            prf=ContractPerformance.PF,
        )
        filtered_events = []
        for e in result.events:
            if e.event_type == EventType.RR:
                continue
            filtered_events.append(
                ContractEvent(
                    event_type=e.event_type,
                    event_time=e.event_time,
                    payoff=e.payoff,
                    currency=e.currency,
                    state_pre=zero_state,
                    state_post=zero_state,
                    sequence=e.sequence,
                )
            )
        return SimulationHistory(
            events=filtered_events,
            states=[zero_state] * len(filtered_events),
            initial_state=result.initial_state,
            final_state=result.final_state,
        )


def _parse_dcc(dcc_str: str) -> DayCountConvention:
    """Parse day count convention string to enum."""
    mapping = {
        "AA": DayCountConvention.AA,
        "A360": DayCountConvention.A360,
        "A365": DayCountConvention.A365,
        "30E360": DayCountConvention.E30360,
        "30E360ISDA": DayCountConvention.E30360ISDA,
        "30360": DayCountConvention.B30360,
        "BUS252": DayCountConvention.BUS252,
    }
    return mapping.get(dcc_str, DayCountConvention.A365)
