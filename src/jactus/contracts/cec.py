"""Credit Enhancement Collateral (CEC) contract implementation.

This module implements credit enhancement collateral contracts that track
collateral value versus covered contract exposure. Similar to margin accounts,
CEC contracts compare collateral provided by covering contracts against the
exposure from covered contracts.

Key Features:
    - Two sets of contracts: covered and covering
    - Covering contracts provide collateral value
    - Covered contracts represent exposure
    - Compares collateral vs required amount
    - Releases excess or seizes shortfall
    - Credit event triggers evaluation

Example:
    >>> from jactus.contracts import CreditEnhancementCollateralContract
    >>> from jactus.core import ContractAttributes, ActusDateTime
    >>> from jactus.observers import ConstantRiskFactorObserver, MockChildContractObserver
    >>>
    >>> # Create collateral contract for a loan
    >>> attrs = ContractAttributes(
    ...     contract_id="CEC-001",
    ...     contract_type=ContractType.CEC,
    ...     contract_role=ContractRole.RPA,
    ...     status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
    ...     maturity_date=ActusDateTime(2029, 1, 1, 0, 0, 0),
    ...     coverage=1.2,  # 120% collateral requirement
    ...     credit_enhancement_guarantee_extent="NO",  # Notional only
    ...     contract_structure='{"CoveredContract": "LOAN-001", "CoveringContract": "STK-001"}',
    ... )
    >>> rf_obs = ConstantRiskFactorObserver(0.03)
    >>> child_obs = MockChildContractObserver()
    >>> cec = CreditEnhancementCollateralContract(attrs, rf_obs, child_obs)
    >>> cashflows = cec.simulate(rf_obs, child_obs)

References:
    ACTUS Technical Specification v1.1, Section 7.18
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


class CECPayoffFunction(BasePayoffFunction):
    """Payoff function for CEC contracts.

    CEC payoffs represent collateral settlement (return or seizure).
    """

    def calculate_payoff(
        self,
        event_type: EventType,
        state: ContractState,
        attributes: ContractAttributes,
        time: ActusDateTime,
        risk_factor_observer: RiskFactorObserver,
    ) -> jnp.ndarray:
        """Calculate payoff for collateral events.

        Args:
            event_type: Type of event
            state: Current contract state
            attributes: Contract attributes
            time: Event time
            risk_factor_observer: Risk factor observer

        Returns:
            Payoff amount (collateral settlement)
        """
        # All payoffs are calculated in the event schedule
        # based on covered and covering contract states
        return jnp.array(0.0, dtype=jnp.float32)


class CECStateTransitionFunction(BaseStateTransitionFunction):
    """State transition function for CEC contracts.

    CEC state tracks collateral value vs exposure.
    """

    def transition_state(
        self,
        event_type: EventType,
        state_pre: ContractState,
        attributes: ContractAttributes,
        time: ActusDateTime,
        risk_factor_observer: RiskFactorObserver,
    ) -> ContractState:
        """Calculate state transition for collateral events.

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


class CreditEnhancementCollateralContract(BaseContract):
    """Credit Enhancement Collateral (CEC) contract.

    A collateral contract that tracks collateral value from covering contracts
    against exposure from covered contracts. When credit events occur or at
    maturity, compares collateral vs required amount and settles appropriately.

    Attributes:
        attributes: Contract terms and conditions
        risk_factor_observer: Observer for market rates
        child_contract_observer: Observer for covered/covering contracts (required)
    """

    def __init__(
        self,
        attributes: ContractAttributes,
        risk_factor_observer: RiskFactorObserver,
        child_contract_observer: ChildContractObserver | None = None,
    ):
        """Initialize CEC contract.

        Args:
            attributes: Contract attributes
            risk_factor_observer: Observer for market data
            child_contract_observer: Observer for covered/covering contracts (required)

        Raises:
            ValueError: If required attributes are missing or invalid
        """
        # Validate contract type
        if attributes.contract_type != ContractType.CEC:
            raise ValueError(f"Expected contract_type=CEC, got {attributes.contract_type}")

        # Validate child contract observer is provided
        if child_contract_observer is None:
            raise ValueError("child_contract_observer is required for CEC contracts")

        # Validate contract structure contains both covered and covering references
        if attributes.contract_structure is None:
            raise ValueError(
                "contract_structure (CTST) is required and must contain "
                "CoveredContract and CoveringContract references"
            )

        # Parse contract structure (JSON string)
        try:
            ctst = json.loads(attributes.contract_structure)
        except (json.JSONDecodeError, TypeError) as e:
            raise ValueError(f"contract_structure must be valid JSON: {e}") from e

        if not isinstance(ctst, dict):
            raise ValueError("contract_structure must be a JSON object (dictionary)")

        # Check for both covered and covering contracts
        has_covered = "CoveredContract" in ctst or "CoveredContracts" in ctst
        has_covering = "CoveringContract" in ctst or "CoveringContracts" in ctst

        if not has_covered:
            raise ValueError(
                "contract_structure must contain 'CoveredContract' or 'CoveredContracts' key"
            )

        if not has_covering:
            raise ValueError(
                "contract_structure must contain 'CoveringContract' or 'CoveringContracts' key"
            )

        # Validate coverage amount
        if attributes.coverage is None:
            raise ValueError("coverage (CECV) is required")

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
            Dictionary with CoveredContract and CoveringContract keys
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

    def _get_covering_contract_ids(self) -> list[str]:
        """Get list of covering contract IDs.

        Returns:
            List of covering contract IDs
        """
        ctst = self._parse_contract_structure()

        # Handle single or multiple covering contracts
        if "CoveringContract" in ctst:
            return [ctst["CoveringContract"]]
        if "CoveringContracts" in ctst:
            contracts = ctst["CoveringContracts"]
            if isinstance(contracts, list):
                return contracts
            if isinstance(contracts, str):
                return [contracts]
            raise ValueError(f"CoveringContracts must be list or string, got {type(contracts)}")

        raise ValueError("contract_structure must contain CoveringContract or CoveringContracts")

    def _calculate_collateral_value(self, time: ActusDateTime) -> float:
        """Calculate total collateral value from covering contracts.

        Args:
            time: Time at which to calculate collateral value

        Returns:
            Total collateral value
        """
        covering_ids = self._get_covering_contract_ids()
        total_value = 0.0

        for contract_id in covering_ids:
            # Query covering contract state
            state = self.child_contract_observer.observe_state(
                contract_id,
                time,
                None,  # State
                None,  # Attributes (child has its own)
            )

            # Use notional as proxy for market value
            # In production, would query actual market value
            value = float(state.nt) if hasattr(state, "nt") else 0.0
            total_value += abs(value)

        return total_value

    def _calculate_exposure(self, time: ActusDateTime) -> float:
        """Calculate total exposure from covered contracts.

        Args:
            time: Time at which to calculate exposure

        Returns:
            Total exposure amount
        """
        covered_ids = self._get_covered_contract_ids()
        cege = self.attributes.credit_enhancement_guarantee_extent
        total_exposure = 0.0

        for contract_id in covered_ids:
            # Query covered contract state
            state = self.child_contract_observer.observe_state(
                contract_id,
                time,
                None,  # State
                None,  # Attributes
            )

            # Calculate exposure based on CEGE mode
            if cege == "NO":
                # Notional only
                exposure = float(state.nt) if hasattr(state, "nt") else 0.0
            elif cege == "NI":
                # Notional + interest
                nt = float(state.nt) if hasattr(state, "nt") else 0.0
                ipac = float(state.ipac) if hasattr(state, "ipac") else 0.0
                exposure = nt + ipac
            elif cege == "MV":
                # Market value (approximated as notional for now)
                exposure = float(state.nt) if hasattr(state, "nt") else 0.0
            else:
                exposure = 0.0

            total_exposure += abs(exposure)

        return total_exposure

    def _check_collateral_sufficiency(self, time: ActusDateTime) -> tuple[bool, float]:
        """Check if collateral is sufficient to cover exposure.

        Args:
            time: Time at which to check sufficiency

        Returns:
            Tuple of (is_sufficient, shortfall_or_excess)
                - is_sufficient: True if collateral >= required
                - shortfall_or_excess: Negative if shortfall, positive if excess
        """
        collateral_value = self._calculate_collateral_value(time)
        exposure = self._calculate_exposure(time)
        coverage_ratio = float(self.attributes.coverage)

        required_collateral = coverage_ratio * exposure
        difference = collateral_value - required_collateral

        return (difference >= 0, difference)

    def generate_event_schedule(self) -> EventSchedule:
        """Generate event schedule for CEC contract.

        The schedule includes:
        1. Analysis dates (AD) if specified
        2. Credit event detection (XD) if covered contract defaults
        3. Settlement event (STD) after credit event or at maturity
        4. Maturity event (MD) if no credit event occurs

        Returns:
            EventSchedule with collateral events
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
            # At maturity, settle collateral
            is_sufficient, difference = self._check_collateral_sufficiency(
                self.attributes.maturity_date
            )

            # Generate STD event for settlement
            # Positive payoff = return excess collateral
            # Negative payoff = seize shortfall
            settlement_amount = min(
                self._calculate_collateral_value(self.attributes.maturity_date),
                self.attributes.coverage * self._calculate_exposure(self.attributes.maturity_date),
            )

            events.append(
                ContractEvent(
                    event_type=EventType.STD,
                    event_time=self.attributes.maturity_date,
                    payoff=settlement_amount,
                    currency=self.attributes.currency or "USD",
                )
            )

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

        State includes collateral value compared to required amount.

        Returns:
            Initial ContractState
        """
        # Calculate initial collateral value and exposure
        collateral_value = self._calculate_collateral_value(self.attributes.status_date)
        exposure = self._calculate_exposure(self.attributes.status_date)
        coverage_ratio = float(self.attributes.coverage)

        # Nt = min(collateral_value, CECV × exposure)
        required_collateral = coverage_ratio * exposure
        nt = min(collateral_value, required_collateral)

        return ContractState(
            tmd=self.attributes.maturity_date or self.attributes.status_date,
            sd=self.attributes.status_date,
            nt=jnp.array(nt, dtype=jnp.float32),
            ipnr=jnp.array(0.0, dtype=jnp.float32),
            ipac=jnp.array(0.0, dtype=jnp.float32),
            feac=jnp.array(0.0, dtype=jnp.float32),
            nsc=jnp.array(1.0, dtype=jnp.float32),
            isc=jnp.array(1.0, dtype=jnp.float32),
            prf=self.attributes.contract_performance or ContractPerformance.PF,
        )

    def get_payoff_function(self, event_type: Any) -> CECPayoffFunction:
        """Get payoff function for CEC contract.

        Args:
            event_type: Type of event (not used, kept for interface compatibility)

        Returns:
            CECPayoffFunction instance
        """
        return CECPayoffFunction(
            contract_role=self.attributes.contract_role,
            currency=self.attributes.currency,
        )

    def get_state_transition_function(self, event_type: Any) -> CECStateTransitionFunction:
        """Get state transition function for CEC contract.

        Args:
            event_type: Type of event (not used, kept for interface compatibility)

        Returns:
            CECStateTransitionFunction instance
        """
        return CECStateTransitionFunction()
