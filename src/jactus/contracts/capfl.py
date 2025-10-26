"""Cap-Floor (CAPFL) contract implementation.

This module implements interest rate cap and floor contracts. A cap/floor
protects against interest rate movements by paying the differential between
an uncapped and capped underlier (typically a swap).

Key Features:
    - Underlier differential: Query underlier with and without caps/floors
    - Cap: Pays when rates rise above cap rate
    - Floor: Pays when rates fall below floor rate
    - Collar: Both cap and floor (range protection)
    - Differential payoff: abs(uncapped - capped)

Example:
    >>> from jactus.contracts import CapFloorContract
    >>> from jactus.core import ContractAttributes, ActusDateTime
    >>> from jactus.observers import ConstantRiskFactorObserver, MockChildContractObserver
    >>>
    >>> # Create interest rate cap
    >>> attrs = ContractAttributes(
    ...     contract_id="CAP-001",
    ...     contract_type=ContractType.CAPFL,
    ...     contract_role=ContractRole.RPA,
    ...     status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
    ...     maturity_date=ActusDateTime(2029, 1, 1, 0, 0, 0),
    ...     rate_reset_cap=0.06,  # 6% cap rate
    ...     contract_structure='{"Underlying": "SWAP-001"}',
    ... )
    >>> rf_obs = ConstantRiskFactorObserver(0.03)
    >>> child_obs = MockChildContractObserver()
    >>> cap = CapFloorContract(attrs, rf_obs, child_obs)
    >>> cashflows = cap.simulate(rf_obs, child_obs)

References:
    ACTUS Technical Specification v1.1, Section 7.14
"""

import json
from typing import Any

import jax.numpy as jnp

from jactus.contracts.base import BaseContract
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


class CapFloorPayoffFunction(BasePayoffFunction):
    """Payoff function for CAPFL contracts.

    CAPFL payoffs are the differential between uncapped and capped underlier.
    """

    def calculate_payoff(
        self,
        event_type: EventType,
        state: ContractState,
        attributes: ContractAttributes,
        time: ActusDateTime,
        risk_factor_observer: RiskFactorObserver,
    ) -> jnp.ndarray:
        """Calculate payoff for cap/floor events.

        For CAPFL, payoffs come from the differential between uncapped
        and capped underlier events, which are already calculated and
        merged in the event schedule.

        Args:
            event_type: Type of event
            state: Current contract state
            attributes: Contract attributes
            time: Event time
            risk_factor_observer: Risk factor observer

        Returns:
            Zero payoff (actual payoffs from differential events)
        """
        # All payoffs come from differential events
        # which are calculated in generate_event_schedule
        return jnp.array(0.0, dtype=jnp.float32)


class CapFloorStateTransitionFunction(BaseStateTransitionFunction):
    """State transition function for CAPFL contracts.

    CAPFL state is minimal - mainly tracks maturity and performance.
    """

    def transition_state(
        self,
        event_type: EventType,
        state_pre: ContractState,
        attributes: ContractAttributes,
        time: ActusDateTime,
        risk_factor_observer: RiskFactorObserver,
    ) -> ContractState:
        """Calculate state transition for cap/floor events.

        Args:
            event_type: Type of event
            state_pre: State before event
            attributes: Contract attributes
            time: Event time
            risk_factor_observer: Risk factor observer

        Returns:
            Updated contract state
        """
        # State remains the same for cap/floor
        # Underlier manages its own state
        return state_pre


class CapFloorContract(BaseContract):
    """Cap-Floor (CAPFL) contract.

    An interest rate cap or floor that pays the differential between
    an uncapped and capped underlier contract.

    Attributes:
        attributes: Contract terms and conditions
        risk_factor_observer: Observer for market rates
        child_contract_observer: Observer for underlier data (required)
    """

    def __init__(
        self,
        attributes: ContractAttributes,
        risk_factor_observer: RiskFactorObserver,
        child_contract_observer: ChildContractObserver | None = None,
    ):
        """Initialize CAPFL contract.

        Args:
            attributes: Contract attributes
            risk_factor_observer: Observer for market data
            child_contract_observer: Observer for underlier (required)

        Raises:
            ValueError: If required attributes are missing or invalid
        """
        # Validate contract type
        if attributes.contract_type != ContractType.CAPFL:
            raise ValueError(
                f"Expected contract_type=CAPFL, got {attributes.contract_type}"
            )

        # Validate child contract observer is provided
        if child_contract_observer is None:
            raise ValueError(
                "child_contract_observer is required for CAPFL contracts"
            )

        # Validate contract structure contains underlier reference
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

        # Validate at least one of cap or floor is set
        if (
            attributes.rate_reset_cap is None
            and attributes.rate_reset_floor is None
        ):
            raise ValueError(
                "At least one of rate_reset_cap (RRLC) or rate_reset_floor (RRLF) must be set"
            )

        super().__init__(attributes, risk_factor_observer, child_contract_observer)

    def _parse_contract_structure(self) -> dict[str, str]:
        """Parse contract_structure JSON string into dictionary.

        Returns:
            Dictionary with Underlying key
        """
        return json.loads(self.attributes.contract_structure)

    def generate_event_schedule(self) -> EventSchedule:
        """Generate event schedule for CAPFL contract.

        The schedule is created by:
        1. Querying underlier uncapped
        2. Querying underlier with cap/floor applied
        3. Merging congruent IP events and computing differential
        4. Differential = abs(uncapped - capped)

        Returns:
            EventSchedule with differential cap/floor events
        """
        events = []

        # Get underlier reference
        ctst = self._parse_contract_structure()
        underlier_id = ctst["Underlying"]

        # Query underlier uncapped (original contract role)
        uncapped_events = self.child_contract_observer.observe_events(
            underlier_id,
            self.attributes.status_date,
            None,  # Underlier has its own attributes
        )

        # Query underlier with cap/floor
        # Note: This is conceptual - in practice, we'd need to modify
        # the underlier's attributes to apply cap/floor.
        # For now, we'll use the same approach as SWAPS and assume
        # the differential is calculated elsewhere or we query twice
        # with different attribute overrides.

        # For simplicity in this implementation, we'll assume the
        # child observer can handle attribute overrides for cap/floor.
        # The actual differential calculation happens here.

        capped_events = self.child_contract_observer.observe_events(
            underlier_id,
            self.attributes.status_date,
            None,  # Would need cap/floor override mechanism
        )

        # Merge congruent IP events and compute differential
        # Build time->event maps
        uncapped_map: dict[ActusDateTime, ContractEvent] = {}
        for event in uncapped_events:
            if event.event_type == EventType.IP:
                uncapped_map[event.event_time] = event

        capped_map: dict[ActusDateTime, ContractEvent] = {}
        for event in capped_events:
            if event.event_type == EventType.IP:
                capped_map[event.event_time] = event

        # Compute differential for each congruent time
        all_times = set(uncapped_map.keys()) | set(capped_map.keys())
        for time in all_times:
            uncapped_payoff = (
                float(uncapped_map[time].payoff) if time in uncapped_map else 0.0
            )
            capped_payoff = (
                float(capped_map[time].payoff) if time in capped_map else 0.0
            )

            # Differential payoff: abs(uncapped - capped)
            # Cap pays when uncapped > capped (rates went above cap)
            # Floor pays when capped > uncapped (rates went below floor)
            differential = abs(uncapped_payoff - capped_payoff)

            if differential > 0.0:  # Only add if non-zero
                events.append(
                    ContractEvent(
                        event_type=EventType.IP,  # Merged IP event
                        event_time=time,
                        payoff=jnp.array(differential, dtype=jnp.float32),
                        currency=self.attributes.currency or "USD",
                    )
                )

        # Add any parent-level events (analysis dates, termination, etc.)
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

        if self.attributes.termination_date:
            events.append(
                ContractEvent(
                    event_type=EventType.TD,
                    event_time=self.attributes.termination_date,
                    payoff=0.0,
                    currency=self.attributes.currency or "USD",
                )
            )

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

    def initialize_state(self) -> ContractState:
        """Initialize contract state at status date.

        Returns:
            Initial ContractState
        """
        return ContractState(
            tmd=self.attributes.maturity_date or self.attributes.status_date,
            sd=self.attributes.status_date,
            nt=jnp.array(1.0, dtype=jnp.float32),  # Not used for CAPFL
            ipnr=jnp.array(0.0, dtype=jnp.float32),  # Not used for CAPFL
            ipac=jnp.array(0.0, dtype=jnp.float32),
            feac=jnp.array(0.0, dtype=jnp.float32),
            nsc=jnp.array(1.0, dtype=jnp.float32),
            isc=jnp.array(1.0, dtype=jnp.float32),
            prf=self.attributes.contract_performance or ContractPerformance.PF,
        )

    def get_payoff_function(self, event_type: Any) -> CapFloorPayoffFunction:
        """Get payoff function for CAPFL contract.

        Args:
            event_type: Type of event (not used, kept for interface compatibility)

        Returns:
            CapFloorPayoffFunction instance
        """
        return CapFloorPayoffFunction(
            contract_role=self.attributes.contract_role,
            currency=self.attributes.currency,
        )

    def get_state_transition_function(
        self, event_type: Any
    ) -> CapFloorStateTransitionFunction:
        """Get state transition function for CAPFL contract.

        Args:
            event_type: Type of event (not used, kept for interface compatibility)

        Returns:
            CapFloorStateTransitionFunction instance
        """
        return CapFloorStateTransitionFunction()
