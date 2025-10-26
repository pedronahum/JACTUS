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

        # Validate coverage amount
        if attributes.coverage is None:
            raise ValueError("coverage (CECV) is required")

        # Validate credit event type
        if attributes.credit_event_type is None:
            raise ValueError("credit_event_type (CET) is required")

        # Validate credit enhancement guarantee extent
        if attributes.credit_enhancement_guarantee_extent is None:
            raise ValueError(
                "credit_enhancement_guarantee_extent (CEGE) is required (NO, NI, or MV)"
            )

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
