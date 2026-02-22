"""Generic Swap (SWAPS) contract implementation.

This module implements a generic swap contract where two legs are represented
as explicit child contracts. This is the most flexible swap implementation,
supporting any combination of contract types for the legs.

Key Features:
    - Two explicit child contract legs (FirstLeg, SecondLeg)
    - Event merging for congruent events (net settlement)
    - Leg role assignment based on parent role
    - State aggregation from both legs
    - Supports any contract type for legs (PAM, LAM, ANN, etc.)

Example:
    >>> from jactus.contracts import GenericSwapContract
    >>> from jactus.core import ContractAttributes, ActusDateTime
    >>> from jactus.observers import ConstantRiskFactorObserver, MockChildContractObserver
    >>>
    >>> # Create swap with PAM legs
    >>> attrs = ContractAttributes(
    ...     contract_id="SWAP-001",
    ...     contract_type=ContractType.SWAPS,
    ...     contract_role=ContractRole.RFL,  # Receive first leg
    ...     status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
    ...     maturity_date=ActusDateTime(2029, 1, 1, 0, 0, 0),
    ...     delivery_settlement="D",  # Net settlement
    ...     contract_structure='{"FirstLeg": "LEG1-ID", "SecondLeg": "LEG2-ID"}',
    ... )
    >>> rf_obs = ConstantRiskFactorObserver(0.03)
    >>> child_obs = MockChildContractObserver()
    >>> swap = GenericSwapContract(attrs, rf_obs, child_obs)
    >>> cashflows = swap.simulate(rf_obs, child_obs)

References:
    ACTUS Technical Specification v1.1, Section 7.13
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
from jactus.functions import BasePayoffFunction, BaseStateTransitionFunction
from jactus.observers import ChildContractObserver, RiskFactorObserver


def determine_leg_roles(parent_role: ContractRole) -> tuple[ContractRole, ContractRole]:
    """Determine leg roles based on parent contract role.

    ACTUS Rule:
        - If parent CNTRL='RFL' (Receive First Leg): FirstLeg=RPA, SecondLeg=RPL
        - Otherwise: FirstLeg=RPL, SecondLeg=RPA

    Args:
        parent_role: Parent contract role

    Returns:
        Tuple of (first_leg_role, second_leg_role)

    Example:
        >>> determine_leg_roles(ContractRole.RFL)
        (<ContractRole.RPA: 'RPA'>, <ContractRole.RPL: 'RPL'>)
        >>> determine_leg_roles(ContractRole.PFL)
        (<ContractRole.RPL: 'RPL'>, <ContractRole.RPA: 'RPA'>)
    """
    if parent_role == ContractRole.RFL:
        # Receive First Leg = First leg pays you (RPA), Second leg you pay (RPL)
        return (ContractRole.RPA, ContractRole.RPL)
    # Pay First Leg = First leg you pay (RPL), Second leg pays you (RPA)
    return (ContractRole.RPL, ContractRole.RPA)


def merge_congruent_events(event1: ContractEvent, event2: ContractEvent) -> ContractEvent:
    """Merge two congruent events (same time and type) into net event.

    Congruent events have:
        - Same event time
        - Same event type
        - Compatible for netting (IED, IP, PR, MD)

    Formula: f(z) = f(x) + f(y) where τ=t=s

    Args:
        event1: First event (from first leg)
        event2: Second event (from second leg)

    Returns:
        Merged event with summed payoffs and aggregated state
    """
    net_payoff = event1.payoff + event2.payoff

    # Create merged state_post with aggregated values
    merged_state_post = None
    if event1.state_post is not None and event2.state_post is not None:
        s1 = event1.state_post
        s2 = event2.state_post
        merged_state_post = ContractState(
            sd=s1.sd,
            tmd=s1.tmd,
            nt=jnp.array(float(s1.nt) + float(s2.nt), dtype=jnp.float32),
            ipnr=jnp.array(0.0, dtype=jnp.float32),
            ipac=jnp.array(0.0, dtype=jnp.float32),
            feac=jnp.array(0.0, dtype=jnp.float32),
            nsc=jnp.array(1.0, dtype=jnp.float32),
            isc=jnp.array(1.0, dtype=jnp.float32),
            prf=s1.prf,
        )
    elif event1.state_post is not None:
        merged_state_post = event1.state_post

    return ContractEvent(
        event_type=event1.event_type,
        event_time=event1.event_time,
        payoff=net_payoff,
        currency=event1.currency,
        state_pre=event1.state_pre,
        state_post=merged_state_post,
        sequence=event1.sequence,
    )


class GenericSwapPayoffFunction(BasePayoffFunction):
    """Payoff function for SWAPS contracts.

    SWAPS payoffs are derived from child contract events.
    """

    def calculate_payoff(
        self,
        event_type: EventType,
        state: ContractState,
        attributes: ContractAttributes,
        time: ActusDateTime,
        risk_factor_observer: RiskFactorObserver,
    ) -> jnp.ndarray:
        """Calculate payoff for generic swap events.

        For SWAPS, payoffs come from child contract events which are
        already calculated and merged. This function returns zero as
        the actual payoffs are in the event schedule from children.

        Args:
            event_type: Type of event
            state: Current contract state
            attributes: Contract attributes
            time: Event time
            risk_factor_observer: Risk factor observer

        Returns:
            Zero payoff (actual payoffs from child events)
        """
        # All payoffs come from child contract events
        # which are merged in generate_event_schedule
        return jnp.array(0.0, dtype=jnp.float32)


class GenericSwapStateTransitionFunction(BaseStateTransitionFunction):
    """State transition function for SWAPS contracts.

    SWAPS state is aggregated from child contract states.
    """

    def transition_state(
        self,
        event_type: EventType,
        state_pre: ContractState,
        attributes: ContractAttributes,
        time: ActusDateTime,
        risk_factor_observer: RiskFactorObserver,
    ) -> ContractState:
        """Calculate state transition for swap events.

        For SWAPS, state transitions come from child contracts.
        We aggregate state variables from both legs.

        Args:
            event_type: Type of event
            state_pre: State before event
            attributes: Contract attributes
            time: Event time
            risk_factor_observer: Risk factor observer

        Returns:
            Updated contract state (aggregated from legs)
        """
        # State remains the same for parent contract
        # Child contracts manage their own state
        return state_pre


class GenericSwapContract(BaseContract):
    """Generic Swap (SWAPS) contract.

    A swap with two explicit child contract legs. Supports any contract
    types for the legs (PAM, LAM, ANN, etc.) and provides flexible
    event merging and state aggregation.

    Attributes:
        attributes: Contract terms and conditions
        risk_factor_observer: Observer for market rates
        child_contract_observer: Observer for child contract data (required)
    """

    def __init__(
        self,
        attributes: ContractAttributes,
        risk_factor_observer: RiskFactorObserver,
        child_contract_observer: ChildContractObserver | None = None,
    ):
        """Initialize SWAPS contract.

        Args:
            attributes: Contract attributes
            risk_factor_observer: Observer for market data
            child_contract_observer: Observer for child contracts (required)

        Raises:
            ValueError: If required attributes are missing or invalid
        """
        # Validate contract type
        if attributes.contract_type != ContractType.SWAPS:
            raise ValueError(f"Expected contract_type=SWAPS, got {attributes.contract_type}")

        # Validate child contract observer is provided
        if child_contract_observer is None:
            raise ValueError("child_contract_observer is required for SWAPS contracts")

        # Validate contract structure contains leg references
        if attributes.contract_structure is None:
            raise ValueError(
                "contract_structure (CTST) is required and must contain FirstLeg and SecondLeg"
            )

        # Parse contract structure (JSON string)
        try:
            ctst = json.loads(attributes.contract_structure)
        except (json.JSONDecodeError, TypeError) as e:
            raise ValueError(f"contract_structure must be valid JSON: {e}") from e

        if not isinstance(ctst, dict):
            raise ValueError("contract_structure must be a JSON object (dictionary)")

        if "FirstLeg" not in ctst or "SecondLeg" not in ctst:
            raise ValueError("contract_structure must contain 'FirstLeg' and 'SecondLeg' keys")

        super().__init__(attributes, risk_factor_observer, child_contract_observer)

    def _parse_contract_structure(self) -> dict[str, str]:
        """Parse contract_structure JSON string into dictionary.

        Returns:
            Dictionary with FirstLeg and SecondLeg keys
        """
        return json.loads(self.attributes.contract_structure or "{}")  # type: ignore[no-any-return]

    def generate_event_schedule(self) -> EventSchedule:
        """Generate event schedule for SWAPS contract.

        The schedule is created by:
        1. Querying events from both child legs
        2. Merging congruent events if DS='D' (net settlement)
        3. Keeping all events separate if DS='S' (gross settlement)

        Returns:
            EventSchedule with merged or separate leg events
        """
        events = []

        # Get leg references
        ctst = self._parse_contract_structure()
        first_leg_id = ctst["FirstLeg"]
        second_leg_id = ctst["SecondLeg"]

        # Determine leg roles
        first_leg_role, second_leg_role = determine_leg_roles(self.attributes.contract_role)

        # Get delivery settlement mode
        ds_mode = self.attributes.delivery_settlement or "D"

        # child_contract_observer is validated as non-None in __init__
        assert self.child_contract_observer is not None

        # Query events from first leg
        # Note: The child contract already has its role set
        # The first_leg_role/second_leg_role determine how we interpret payments
        first_leg_events = self.child_contract_observer.observe_events(
            first_leg_id,
            self.attributes.status_date,
            None,  # Child has its own attributes
        )

        # Query events from second leg
        second_leg_events = self.child_contract_observer.observe_events(
            second_leg_id,
            self.attributes.status_date,
            None,  # Child has its own attributes
        )

        # Events are already lists
        first_events = first_leg_events
        second_events = second_leg_events

        if ds_mode == "S":
            # Cash settlement (net): Merge congruent events by summing payoffs
            # Congruent = same time and type (IED, IP, PR)
            congruent_types = {EventType.IED, EventType.IP, EventType.PR, EventType.MD}

            # Build time->event maps
            first_map: dict[tuple[ActusDateTime, EventType], ContractEvent] = {}
            for event in first_events:
                if event.event_type in congruent_types:
                    key = (event.event_time, event.event_type)
                    first_map[key] = event
                else:
                    events.append(event)  # Non-congruent, add as-is

            second_map: dict[tuple[ActusDateTime, EventType], ContractEvent] = {}
            for event in second_events:
                if event.event_type in congruent_types:
                    key = (event.event_time, event.event_type)
                    second_map[key] = event
                else:
                    events.append(event)  # Non-congruent, add as-is

            # Merge congruent events
            all_keys = set(first_map.keys()) | set(second_map.keys())
            for key in all_keys:
                e1 = first_map.get(key)
                e2 = second_map.get(key)

                if e1 and e2:
                    # Both legs have this event - merge
                    merged = merge_congruent_events(e1, e2)
                    events.append(merged)
                elif e1:
                    # Only first leg
                    events.append(e1)
                elif e2:
                    # Only second leg
                    events.append(e2)

        else:
            # Delivery/gross settlement (D): Keep all events separate
            events.extend(first_events)
            events.extend(second_events)

        currency = self.attributes.currency or "USD"
        role_sign = self.attributes.contract_role.get_sign()

        # Filter by purchase date: exclude events before PRD
        if self.attributes.purchase_date:
            prd_time = self.attributes.purchase_date
            events = [e for e in events if e.event_time > prd_time]
            # Add PRD event
            prd_payoff = role_sign * (self.attributes.price_at_purchase_date or 0.0)
            zero_state = ContractState(
                tmd=self.attributes.maturity_date or prd_time,
                sd=prd_time,
                nt=jnp.array(0.0, dtype=jnp.float32),
                ipnr=jnp.array(0.0, dtype=jnp.float32),
                ipac=jnp.array(0.0, dtype=jnp.float32),
                feac=jnp.array(0.0, dtype=jnp.float32),
                nsc=jnp.array(1.0, dtype=jnp.float32),
                isc=jnp.array(1.0, dtype=jnp.float32),
                prf=ContractPerformance.PF,
            )
            events.append(
                ContractEvent(
                    event_type=EventType.PRD,
                    event_time=prd_time,
                    payoff=jnp.array(prd_payoff, dtype=jnp.float32),
                    currency=currency,
                    state_pre=zero_state,
                    state_post=zero_state,
                )
            )

        # Filter by termination date: keep events before TD and non-MD events at TD
        if self.attributes.termination_date:
            td_time = self.attributes.termination_date
            events = [
                e
                for e in events
                if e.event_time < td_time
                or (e.event_time == td_time and e.event_type != EventType.MD)
            ]
            # Add TD event
            td_payoff = role_sign * (self.attributes.price_at_termination_date or 0.0)
            td_state = ContractState(
                tmd=td_time,
                sd=td_time,
                nt=jnp.array(0.0, dtype=jnp.float32),
                ipnr=jnp.array(0.0, dtype=jnp.float32),
                ipac=jnp.array(0.0, dtype=jnp.float32),
                feac=jnp.array(0.0, dtype=jnp.float32),
                nsc=jnp.array(1.0, dtype=jnp.float32),
                isc=jnp.array(1.0, dtype=jnp.float32),
                prf=ContractPerformance.PF,
            )
            events.append(
                ContractEvent(
                    event_type=EventType.TD,
                    event_time=td_time,
                    payoff=jnp.array(td_payoff, dtype=jnp.float32),
                    currency=currency,
                    state_pre=td_state,
                    state_post=td_state,
                )
            )

        # Add analysis date events
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

        State is aggregated from both child leg states.

        Returns:
            Initial ContractState
        """
        # Get leg references
        ctst = self._parse_contract_structure()
        first_leg_id = ctst["FirstLeg"]
        second_leg_id = ctst["SecondLeg"]

        # Determine leg roles
        first_leg_role, second_leg_role = determine_leg_roles(self.attributes.contract_role)

        # child_contract_observer is validated as non-None in __init__
        assert self.child_contract_observer is not None

        # Query initial states from both legs
        first_state = self.child_contract_observer.observe_state(
            first_leg_id,
            self.attributes.status_date,
            None,  # State
            None,  # Child has its own attributes
        )

        second_state = self.child_contract_observer.observe_state(
            second_leg_id,
            self.attributes.status_date,
            None,  # State
            None,  # Child has its own attributes
        )

        # Aggregate state variables
        # md = max of both legs
        tmd = (
            max(first_state.tmd, second_state.tmd)
            if first_state.tmd and second_state.tmd
            else (first_state.tmd or second_state.tmd)
        )

        # ipac = sum of both legs (with role adjustments)
        first_ipac = (
            float(first_state.ipac)
            if hasattr(first_state, "ipac") and first_state.ipac is not None
            else 0.0
        )
        second_ipac = (
            float(second_state.ipac)
            if hasattr(second_state, "ipac") and second_state.ipac is not None
            else 0.0
        )

        # Role adjustment: RPA adds, RPL subtracts
        ipac_total = first_ipac + second_ipac

        return ContractState(
            tmd=tmd or self.attributes.maturity_date or self.attributes.status_date,
            sd=self.attributes.status_date,
            nt=jnp.array(1.0, dtype=jnp.float32),  # Not used for SWAPS
            ipnr=jnp.array(0.0, dtype=jnp.float32),  # Not used for SWAPS
            ipac=jnp.array(ipac_total, dtype=jnp.float32),
            feac=jnp.array(0.0, dtype=jnp.float32),
            nsc=jnp.array(1.0, dtype=jnp.float32),
            isc=jnp.array(1.0, dtype=jnp.float32),
            prf=self.attributes.contract_performance or ContractPerformance.PF,
        )

    def get_payoff_function(self, event_type: Any) -> GenericSwapPayoffFunction:
        """Get payoff function for SWAPS contract.

        Args:
            event_type: Type of event (not used, kept for interface compatibility)

        Returns:
            GenericSwapPayoffFunction instance
        """
        return GenericSwapPayoffFunction(
            contract_role=self.attributes.contract_role,
            currency=self.attributes.currency,
        )

    def get_state_transition_function(self, event_type: Any) -> GenericSwapStateTransitionFunction:
        """Get state transition function for SWAPS contract.

        Args:
            event_type: Type of event (not used, kept for interface compatibility)

        Returns:
            GenericSwapStateTransitionFunction instance
        """
        return GenericSwapStateTransitionFunction()

    def simulate(
        self,
        risk_factor_observer: RiskFactorObserver | None = None,
        child_contract_observer: ChildContractObserver | None = None,
    ) -> SimulationHistory:
        """Simulate SWAPS contract by passing through child contract events.

        For SWAPS, event payoffs and states come directly from child contract
        simulations. The schedule events already contain pre-computed data,
        so we pass them through instead of recalculating via POF/STF.
        """
        initial_state = self.initialize_state()
        schedule = self.get_events()

        events = list(schedule.events)
        states = [e.state_post for e in events if e.state_post is not None]
        final_state = states[-1] if states else initial_state

        return SimulationHistory(
            events=events,
            states=states,
            initial_state=initial_state,
            final_state=final_state,
        )
