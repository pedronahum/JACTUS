"""Credit Enhancement Guarantee (CEG) contract implementation.

This module implements credit enhancement guarantee contracts that cover losses
on covered contracts when credit events occur. Similar to credit default swaps,
CEG contracts pay out when the performance of covered contracts deteriorates
to a specified credit event type.

Key Features:
    - Covers losses on one or more contracts
    - Credit event triggers payout
    - Coverage amount calculated from covered contracts
    - Guarantee fees paid periodically
    - Multiple coverage extent modes (NO, NI, MV)

Example:
    >>> from jactus.contracts import CreditEnhancementGuaranteeContract
    >>> from jactus.core import ContractAttributes, ActusDateTime
    >>> from jactus.observers import ConstantRiskFactorObserver, MockChildContractObserver
    >>>
    >>> # Create credit guarantee covering a loan
    >>> attrs = ContractAttributes(
    ...     contract_id="CEG-001",
    ...     contract_type=ContractType.CEG,
    ...     contract_role=ContractRole.RPA,
    ...     status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
    ...     maturity_date=ActusDateTime(2029, 1, 1, 0, 0, 0),
    ...     coverage=0.8,  # 80% coverage
    ...     credit_event_type=ContractPerformance.DL,  # Default
    ...     credit_enhancement_guarantee_extent="NO",  # Notional only
    ...     contract_structure='{"CoveredContract": "LOAN-001"}',
    ...     fee_rate=0.01,  # 1% annual fee
    ...     fee_payment_cycle="P1Y",
    ... )
    >>> rf_obs = ConstantRiskFactorObserver(0.03)
    >>> child_obs = MockChildContractObserver()
    >>> ceg = CreditEnhancementGuaranteeContract(attrs, rf_obs, child_obs)
    >>> cashflows = ceg.simulate(rf_obs, child_obs)

References:
    ACTUS Technical Specification v1.1, Section 7.17
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
    ContractState,
    ContractType,
    EventSchedule,
    EventType,
)
from jactus.functions import BasePayoffFunction, BaseStateTransitionFunction
from jactus.observers import ChildContractObserver, RiskFactorObserver
from jactus.utilities.schedules import generate_schedule


class CEGPayoffFunction(BasePayoffFunction):
    """Payoff function for CEG contracts.

    CEG payoffs include guarantee fees and credit event payouts.
    """

    def calculate_payoff(
        self,
        event_type: EventType,
        state: ContractState,
        attributes: ContractAttributes,
        time: ActusDateTime,
        risk_factor_observer: RiskFactorObserver,
    ) -> jnp.ndarray:
        """Calculate payoff for credit guarantee events.

        Args:
            event_type: Type of event
            state: Current contract state
            attributes: Contract attributes
            time: Event time
            risk_factor_observer: Risk factor observer

        Returns:
            Payoff amount (fees or credit event payout)
        """
        # All payoffs are calculated in the event schedule
        # based on covered contract states
        return jnp.array(0.0, dtype=jnp.float32)


class CEGStateTransitionFunction(BaseStateTransitionFunction):
    """State transition function for CEG contracts.

    CEG state tracks coverage amount, fee accrual, and exercise status.
    """

    def transition_state(
        self,
        event_type: EventType,
        state_pre: ContractState,
        attributes: ContractAttributes,
        time: ActusDateTime,
        risk_factor_observer: RiskFactorObserver,
    ) -> ContractState:
        """Calculate state transition for guarantee events.

        Args:
            event_type: Type of event
            state_pre: State before event
            attributes: Contract attributes
            time: Event time
            risk_factor_observer: Risk factor observer

        Returns:
            Updated contract state
        """
        # State updates handled per event type
        # Most state is in the child contracts
        return ContractState(
            tmd=state_pre.tmd,
            sd=time,
            nt=state_pre.nt,
            ipnr=state_pre.ipnr,
            ipac=state_pre.ipac,
            feac=state_pre.feac,
            nsc=state_pre.nsc,
            isc=state_pre.isc,
            prf=state_pre.prf,
        )


class CreditEnhancementGuaranteeContract(BaseContract):
    """Credit Enhancement Guarantee (CEG) contract.

    A guarantee contract that pays out when covered contracts experience
    credit events. The payout covers a specified percentage of the covered
    amount, calculated based on the coverage extent mode (notional only,
    notional plus interest, or market value).

    Attributes:
        attributes: Contract terms and conditions
        risk_factor_observer: Observer for market rates
        child_contract_observer: Observer for covered contract data (required)
    """

    def __init__(
        self,
        attributes: ContractAttributes,
        risk_factor_observer: RiskFactorObserver,
        child_contract_observer: ChildContractObserver | None = None,
    ):
        """Initialize CEG contract.

        Args:
            attributes: Contract attributes
            risk_factor_observer: Observer for market data
            child_contract_observer: Observer for covered contracts (required)

        Raises:
            ValueError: If required attributes are missing or invalid
        """
        # Validate contract type
        if attributes.contract_type != ContractType.CEG:
            raise ValueError(f"Expected contract_type=CEG, got {attributes.contract_type}")

        # Validate child contract observer is provided
        if child_contract_observer is None:
            raise ValueError("child_contract_observer is required for CEG contracts")

        # Validate contract structure contains covered contract references
        if attributes.contract_structure is None:
            raise ValueError(
                "contract_structure (CTST) is required and must contain CoveredContract reference(s)"
            )

        # Parse contract structure (JSON string)
        try:
            ctst = json.loads(attributes.contract_structure)
        except (json.JSONDecodeError, TypeError) as e:
            raise ValueError(f"contract_structure must be valid JSON: {e}") from e

        if not isinstance(ctst, dict):
            raise ValueError("contract_structure must be a JSON object (dictionary)")

        if "CoveredContract" not in ctst and "CoveredContracts" not in ctst:
            raise ValueError(
                "contract_structure must contain 'CoveredContract' or 'CoveredContracts' key"
            )

        # Default coverage to 1.0 (full coverage) if not specified
        if attributes.coverage is None:
            attributes.coverage = 1.0

        # Default credit event type to "DF" (default) if not specified
        if attributes.credit_event_type is None:
            attributes.credit_event_type = "DF"

        # Default guarantee extent to "NO" (notional only) if not specified
        if attributes.credit_enhancement_guarantee_extent is None:
            attributes.credit_enhancement_guarantee_extent = "NO"

        if attributes.credit_enhancement_guarantee_extent not in ["NO", "NI", "MV"]:
            raise ValueError(
                f"credit_enhancement_guarantee_extent must be NO, NI, or MV, "
                f"got {attributes.credit_enhancement_guarantee_extent}"
            )

        super().__init__(attributes, risk_factor_observer, child_contract_observer)

    def _parse_contract_structure(self) -> dict[str, Any]:
        """Parse contract_structure JSON string into dictionary.

        Returns:
            Dictionary with CoveredContract or CoveredContracts key
        """
        return json.loads(self.attributes.contract_structure)

    def _get_covered_contract_ids(self) -> list[str]:
        """Get list of covered contract IDs.

        Returns:
            List of covered contract IDs
        """
        ctst = self._parse_contract_structure()

        # Handle single or multiple covered contracts
        if "CoveredContract" in ctst:
            return [ctst["CoveredContract"]]
        if "CoveredContracts" in ctst:
            contracts = ctst["CoveredContracts"]
            if isinstance(contracts, list):
                return contracts
            if isinstance(contracts, str):
                return [contracts]
            raise ValueError(f"CoveredContracts must be list or string, got {type(contracts)}")
        raise ValueError("contract_structure must contain CoveredContract or CoveredContracts")

    def _calculate_coverage_amount(self, time: ActusDateTime) -> float:
        """Calculate total coverage amount for all covered contracts.

        Args:
            time: Time at which to calculate coverage

        Returns:
            Total coverage amount
        """
        covered_ids = self._get_covered_contract_ids()
        cege = self.attributes.credit_enhancement_guarantee_extent
        total_amount = 0.0

        for contract_id in covered_ids:
            # Query covered contract state
            state = self.child_contract_observer.observe_state(
                contract_id,
                time,
                None,  # State
                None,  # Attributes (child has its own)
            )

            # Calculate amount based on CEGE mode
            if cege == "NO":
                # Notional only
                amount = float(state.nt) if hasattr(state, "nt") else 0.0
            elif cege == "NI":
                # Notional + interest
                nt = float(state.nt) if hasattr(state, "nt") else 0.0
                ipac = float(state.ipac) if hasattr(state, "ipac") else 0.0
                amount = nt + ipac
            elif cege == "MV":
                # Market value (approximated as notional for now)
                # In production, would query market value from risk factor observer
                amount = float(state.nt) if hasattr(state, "nt") else 0.0
            else:
                amount = 0.0

            total_amount += abs(amount)  # Use absolute value for coverage

        # Apply coverage ratio
        coverage_ratio = float(self.attributes.coverage)
        return coverage_ratio * total_amount

    def _detect_credit_event(self, time: ActusDateTime) -> bool:
        """Detect if a credit event has occurred on any covered contract.

        Args:
            time: Time at which to check for credit events

        Returns:
            True if credit event detected, False otherwise
        """
        covered_ids = self._get_covered_contract_ids()
        target_performance = self.attributes.credit_event_type

        for contract_id in covered_ids:
            # Query covered contract state
            state = self.child_contract_observer.observe_state(
                contract_id,
                time,
                None,  # State
                None,  # Attributes
            )

            # Check if performance matches credit event type
            if hasattr(state, "prf") and state.prf == target_performance:
                return True

        return False

    def generate_event_schedule(self) -> EventSchedule:
        """Generate event schedule for CEG contract.

        The schedule includes:
        1. Fee payment events (FP) if fees are charged
        2. Credit event detection (XD) if covered contract defaults
        3. Settlement event (STD) after credit event
        4. Maturity event (MD) if no credit event occurs

        Returns:
            EventSchedule with guarantee events
        """
        events = []

        # Add analysis dates if specified
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

        # Add fee payment events if fee schedule is defined
        # Note: FP schedule generation would require schedule utilities
        # For now, we add a single FP event at maturity if fees are specified
        if self.attributes.fee_payment_cycle and self.attributes.maturity_date:
            fee_rate = self.attributes.fee_rate or 0.0

            if fee_rate > 0:
                # Simplified: single fee event at maturity
                # In production, would generate periodic FP events using cycle
                coverage_amount = self._calculate_coverage_amount(self.attributes.maturity_date)
                fee_amount = coverage_amount * fee_rate

                events.append(
                    ContractEvent(
                        event_type=EventType.FP,
                        event_time=self.attributes.maturity_date,
                        payoff=fee_amount,
                        currency=self.attributes.currency or "USD",
                    )
                )

        # Check for credit events (simplified - in production would observe from children)
        # For now, we don't generate XD/STD events as they're event-driven
        # They would be detected during simulation when querying covered contracts

        # Add termination date if specified
        if self.attributes.termination_date:
            events.append(
                ContractEvent(
                    event_type=EventType.TD,
                    event_time=self.attributes.termination_date,
                    payoff=0.0,
                    currency=self.attributes.currency or "USD",
                )
            )

        # Add maturity event
        if self.attributes.maturity_date:
            events.append(
                ContractEvent(
                    event_type=EventType.MD,
                    event_time=self.attributes.maturity_date,
                    payoff=0.0,
                    currency=self.attributes.currency or "USD",
                )
            )

        # Sort events by time
        events.sort(
            key=lambda e: (e.event_time.year, e.event_time.month, e.event_time.day, e.sequence)
        )

        return EventSchedule(
            contract_id=self.attributes.contract_id,
            events=tuple(events),
        )

    def initialize_state(self) -> ContractState:
        """Initialize contract state at status date.

        State includes coverage amount calculated from covered contracts.

        Returns:
            Initial ContractState
        """
        # Calculate initial coverage amount
        coverage_amount = self._calculate_coverage_amount(self.attributes.status_date)

        return ContractState(
            tmd=self.attributes.maturity_date or self.attributes.status_date,
            sd=self.attributes.status_date,
            nt=jnp.array(coverage_amount, dtype=jnp.float32),
            ipnr=jnp.array(0.0, dtype=jnp.float32),
            ipac=jnp.array(0.0, dtype=jnp.float32),
            feac=jnp.array(0.0, dtype=jnp.float32),
            nsc=jnp.array(1.0, dtype=jnp.float32),
            isc=jnp.array(1.0, dtype=jnp.float32),
            prf=self.attributes.contract_performance or ContractPerformance.PF,
        )

    def get_payoff_function(self, event_type: Any) -> CEGPayoffFunction:
        """Get payoff function for CEG contract.

        Args:
            event_type: Type of event (not used, kept for interface compatibility)

        Returns:
            CEGPayoffFunction instance
        """
        return CEGPayoffFunction(
            contract_role=self.attributes.contract_role,
            currency=self.attributes.currency,
        )

    def get_state_transition_function(self, event_type: Any) -> CEGStateTransitionFunction:
        """Get state transition function for CEG contract.

        Args:
            event_type: Type of event (not used, kept for interface compatibility)

        Returns:
            CEGStateTransitionFunction instance
        """
        return CEGStateTransitionFunction()

    def simulate(
        self,
        risk_factor_observer: RiskFactorObserver | None = None,
        child_contract_observer: ChildContractObserver | None = None,
    ) -> SimulationHistory:
        """Simulate CEG contract with comprehensive event generation.

        Generates PRD, FP, XD, STD, and MD events based on covered contract
        states and credit events observed through the child observer.
        """
        from datetime import timedelta

        from jactus.utilities.conventions import year_fraction

        role_sign = self.attributes.contract_role.get_sign()
        currency = self.attributes.currency or "USD"
        events: list[ContractEvent] = []
        covered_ids = self._get_covered_contract_ids()

        # Determine effective maturity (from attributes or inferred from children)
        effective_maturity = self.attributes.maturity_date
        if effective_maturity is None:
            for cid in covered_ids:
                try:
                    child_attrs = self.child_contract_observer._attributes.get(cid)
                    if child_attrs and child_attrs.maturity_date:
                        child_md = child_attrs.maturity_date
                        if effective_maturity is None or child_md > effective_maturity:
                            effective_maturity = child_md
                except (AttributeError, KeyError):
                    pass

        # Calculate coverage at purchase_date (children have been funded by then)
        coverage_time = self.attributes.purchase_date or self.attributes.status_date
        coverage_amount = self._calculate_coverage_with_accrual(coverage_time)
        current_nt = role_sign * coverage_amount

        def _make_state(
            time: ActusDateTime,
            nt: float,
            prf: ContractPerformance = ContractPerformance.PF,
        ) -> ContractState:
            return ContractState(
                tmd=effective_maturity or time,
                sd=time,
                nt=jnp.array(nt, dtype=jnp.float32),
                ipnr=jnp.array(0.0, dtype=jnp.float32),
                ipac=jnp.array(0.0, dtype=jnp.float32),
                feac=jnp.array(0.0, dtype=jnp.float32),
                nsc=jnp.array(1.0, dtype=jnp.float32),
                isc=jnp.array(1.0, dtype=jnp.float32),
                prf=prf,
            )

        exercised = False

        # PRD event at purchase date
        if self.attributes.purchase_date:
            prd_time = self.attributes.purchase_date
            prd_payoff = -role_sign * (self.attributes.price_at_purchase_date or 0.0)
            prd_state = _make_state(prd_time, current_nt)
            events.append(ContractEvent(
                event_type=EventType.PRD,
                event_time=prd_time,
                payoff=jnp.array(prd_payoff, dtype=jnp.float32),
                currency=currency,
                state_pre=prd_state,
                state_post=prd_state,
            ))

        # FP events from fee payment schedule
        if self.attributes.fee_payment_cycle and self.attributes.fee_rate is not None:
            fp_start = (
                self.attributes.fee_payment_anchor
                or self.attributes.purchase_date
                or self.attributes.status_date
            )
            fp_end = effective_maturity or self.attributes.status_date
            cycle = self.attributes.fee_payment_cycle
            fp_dates = generate_schedule(start=fp_start, cycle=cycle, end=fp_end)
            for fp_time in fp_dates:
                if fp_time <= self.attributes.status_date:
                    continue
                # Calculate coverage at FP time for notional tracking
                try:
                    fp_coverage = self._calculate_coverage_with_accrual(fp_time)
                    fp_nt = role_sign * fp_coverage
                except Exception:
                    fp_nt = current_nt
                fp_state = _make_state(fp_time, fp_nt)
                fp_payoff = role_sign * (self.attributes.fee_rate or 0.0)
                events.append(ContractEvent(
                    event_type=EventType.FP,
                    event_time=fp_time,
                    payoff=jnp.array(fp_payoff, dtype=jnp.float32),
                    currency=currency,
                    state_pre=fp_state,
                    state_post=fp_state,
                ))

        # Detect credit events from child observer
        target_perf = self.attributes.credit_event_type or "DF"
        ce_time = None
        exercise_amount = 0.0

        for cid in covered_ids:
            try:
                child_events = self.child_contract_observer.observe_events(
                    cid, self.attributes.status_date, None
                )
            except (KeyError, ValueError):
                continue
            for ce in child_events:
                if ce.event_type == EventType.CE and ce.state_post is not None:
                    # Check performance matches target
                    prf_match = (
                        str(ce.state_post.prf) == target_perf
                        or (hasattr(ce.state_post.prf, "value") and ce.state_post.prf.value == target_perf)
                    )
                    if not prf_match:
                        continue
                    # Only consider CE events before effective maturity
                    if effective_maturity and ce.event_time > effective_maturity:
                        continue
                    ce_time = ce.event_time
                    exercise_amount = self._calculate_coverage_with_accrual(ce_time)
                    break
            if ce_time:
                break

        # XD and STD events if credit event detected
        if ce_time is not None:
            exercised = True
            xd_nt = role_sign * exercise_amount
            xd_state = _make_state(ce_time, xd_nt)
            events.append(ContractEvent(
                event_type=EventType.XD,
                event_time=ce_time,
                payoff=jnp.array(0.0, dtype=jnp.float32),
                currency=currency,
                state_pre=xd_state,
                state_post=xd_state,
            ))

            # STD time = XD + settlementPeriod (with business day adjustment)
            std_time = self._compute_settlement_time(ce_time)

            # STD payoff = exercise amount
            # Add accrued interest only if settlement period is non-zero
            # (business day shifts from P0D don't create an accrual period)
            std_payoff = role_sign * exercise_amount
            raw_delay = self._get_settlement_period_days()
            if raw_delay > 0:
                # Compute accrual using raw settlement end (before bday adjustment)
                raw_end = self._compute_raw_settlement_end(ce_time)
                accrual = self._compute_accrual_between(ce_time, raw_end)
                std_payoff += role_sign * accrual

            std_state = _make_state(std_time, 0.0)
            events.append(ContractEvent(
                event_type=EventType.STD,
                event_time=std_time,
                payoff=jnp.array(std_payoff, dtype=jnp.float32),
                currency=currency,
                state_pre=xd_state,
                state_post=std_state,
            ))

        # MD event at effective maturity (if not exercised)
        if effective_maturity and not exercised:
            md_state = _make_state(effective_maturity, 0.0)
            events.append(ContractEvent(
                event_type=EventType.MD,
                event_time=effective_maturity,
                payoff=jnp.array(0.0, dtype=jnp.float32),
                currency=currency,
                state_pre=_make_state(effective_maturity, current_nt),
                state_post=md_state,
            ))

        # Sort events
        events.sort(key=lambda e: (
            e.event_time.year, e.event_time.month, e.event_time.day, e.sequence
        ))

        # Filter out FP events at or after STD if exercised
        if exercised and ce_time:
            std_time = self._compute_settlement_time(ce_time)
            events = [
                e for e in events
                if e.event_time < std_time
                or e.event_type in (EventType.XD, EventType.STD)
            ]

        initial_state = _make_state(self.attributes.status_date, current_nt)
        states = [e.state_post for e in events if e.state_post is not None]
        final_state = states[-1] if states else initial_state

        return SimulationHistory(
            events=events,
            states=states,
            initial_state=initial_state,
            final_state=final_state,
        )

    def _get_child_dcc(self, child_id: str):
        """Get day count convention for a child contract."""
        try:
            child_attrs = self.child_contract_observer._attributes.get(child_id)
            if child_attrs and child_attrs.day_count_convention:
                return child_attrs.day_count_convention
        except (AttributeError, KeyError):
            pass
        from jactus.core.types import DayCountConvention
        return DayCountConvention.A365

    def _calculate_coverage_with_accrual(self, time: ActusDateTime) -> float:
        """Calculate coverage amount with proper accrued interest for NI mode.

        For NO mode: sum of abs(nt) * coverage
        For NI mode: sum of (abs(nt) + accrued_interest) * coverage
        For MV mode: same as NO (approximation)
        """
        from jactus.utilities.conventions import year_fraction

        covered_ids = self._get_covered_contract_ids()
        cege = self.attributes.credit_enhancement_guarantee_extent
        coverage_ratio = float(self.attributes.coverage)
        total = 0.0

        for cid in covered_ids:
            try:
                state = self.child_contract_observer.observe_state(cid, time, None, None)
            except (KeyError, ValueError):
                continue
            nt = abs(float(state.nt))

            if cege == "NI":
                ipnr = abs(float(state.ipnr)) if state.ipnr is not None else 0.0
                ipac = abs(float(state.ipac)) if state.ipac is not None else 0.0
                dcc = self._get_child_dcc(cid)
                yf = year_fraction(state.sd, time, dcc)
                accrued = ipac + yf * ipnr * nt
                total += nt + accrued
            else:
                # NO or MV mode: notional only
                total += nt

        return coverage_ratio * total

    def _compute_accrual_between(
        self, start: ActusDateTime, end: ActusDateTime,
    ) -> float:
        """Compute total accrued interest on covered contracts between two times."""
        from jactus.utilities.conventions import year_fraction

        covered_ids = self._get_covered_contract_ids()
        total = 0.0
        for cid in covered_ids:
            try:
                state = self.child_contract_observer.observe_state(cid, start, None, None)
            except (KeyError, ValueError):
                continue
            nt = abs(float(state.nt))
            ipnr = abs(float(state.ipnr)) if state.ipnr is not None else 0.0
            dcc = self._get_child_dcc(cid)
            yf = year_fraction(start, end, dcc)
            total += yf * ipnr * nt
        return total

    def _get_settlement_period_days(self) -> int:
        """Parse settlement period and return the number of days (0 for P0D)."""
        import re
        sp = self.attributes.settlement_period
        if not sp:
            return 0
        sp = sp[1:] if sp.startswith("P") else sp
        if "L" in sp:
            sp = sp[:sp.index("L")]
        m = re.match(r"(\d+)([DWMY])", sp)
        if not m:
            return 0
        n, unit = int(m.group(1)), m.group(2)
        if unit == "D":
            return n
        if unit == "W":
            return n * 7
        # For M/Y, approximate
        return n * 30 if unit == "M" else n * 365

    def _compute_raw_settlement_end(self, xd_time: ActusDateTime) -> ActusDateTime:
        """Compute settlement end date without business day adjustment."""
        import re
        sp = self.attributes.settlement_period
        if not sp:
            return xd_time
        sp_str = sp[1:] if sp.startswith("P") else sp
        if "L" in sp_str:
            sp_str = sp_str[:sp_str.index("L")]
        m = re.match(r"(\d+)([DWMY])", sp_str)
        if not m:
            return xd_time
        n, unit = int(m.group(1)), m.group(2)
        if n == 0:
            return xd_time
        from dateutil.relativedelta import relativedelta
        xd_py = xd_time.to_datetime()
        delta_map = {
            "D": relativedelta(days=n),
            "W": relativedelta(weeks=n),
            "M": relativedelta(months=n),
            "Y": relativedelta(years=n),
        }
        end_py = xd_py + delta_map.get(unit, relativedelta())
        return ActusDateTime(end_py.year, end_py.month, end_py.day, end_py.hour, end_py.minute, end_py.second)

    def _adjust_business_day(self, time: ActusDateTime) -> ActusDateTime:
        """Adjust date to next business day if it falls on a weekend.

        Only applies when the contract has a non-trivial calendar (e.g., MF).
        """
        from jactus.core.types import Calendar
        calendar = self.attributes.calendar
        if calendar in (Calendar.NO_CALENDAR, None):
            return time
        dt = time.to_datetime()
        while dt.weekday() >= 5:  # Saturday=5, Sunday=6
            from datetime import timedelta
            dt += timedelta(days=1)
        return ActusDateTime(dt.year, dt.month, dt.day, dt.hour, dt.minute, dt.second)

    def _compute_settlement_time(self, xd_time: ActusDateTime) -> ActusDateTime:
        """Compute settlement time from exercise time + settlement period."""
        settlement_period = self.attributes.settlement_period
        if not settlement_period:
            return self._adjust_business_day(xd_time)

        # Parse settlement period (e.g., "P0D", "P5DL0")
        import re
        sp = settlement_period
        if sp.startswith("P"):
            sp = sp[1:]
        if "L" in sp:
            sp = sp[:sp.index("L")]
        m = re.match(r"(\d+)([DWMY])", sp)
        if not m:
            return self._adjust_business_day(xd_time)

        n, unit = int(m.group(1)), m.group(2)
        if n == 0:
            return self._adjust_business_day(xd_time)

        from dateutil.relativedelta import relativedelta
        xd_py = xd_time.to_datetime()
        delta_map = {
            "D": relativedelta(days=n),
            "W": relativedelta(weeks=n),
            "M": relativedelta(months=n),
            "Y": relativedelta(years=n),
        }
        std_py = xd_py + delta_map.get(unit, relativedelta())
        result = ActusDateTime(
            std_py.year, std_py.month, std_py.day,
            std_py.hour, std_py.minute, std_py.second,
        )
        return self._adjust_business_day(result)
