"""Contract simulation engine for JACTUS.

This module provides the core simulation engine for running ACTUS contracts through
their lifecycles and generating cashflow projections.

Key Features:
- Single contract simulation
- Scenario-based simulation (what-if analysis)
- Batch simulation with JAX vmap
- Cashflow timeline generation
- Rich simulation results with history

ACTUS References:
    ACTUS v1.1 Section 4.5 - Contract Simulation
    ACTUS v1.1 Section 4.6 - Scenario Analysis

Example:
    >>> from jactus.engine import ContractSimulator, SimulationResult
    >>> from jactus.contracts import PAMContract
    >>> from jactus.observers import ConstantRiskFactorObserver
    >>>
    >>> # Create simulator
    >>> simulator = ContractSimulator()
    >>>
    >>> # Simulate single contract
    >>> contract = PAMContract(attributes=..., risk_factor_observer=...)
    >>> result = simulator.simulate_contract(contract)
    >>>
    >>> # Access cashflows
    >>> df = result.to_dataframe()
    >>> timeline = result.get_cashflow_timeline()
    >>> npv = result.calculate_npv(discount_rate=0.05)
"""

from dataclasses import dataclass, field

# Import TYPE_CHECKING to avoid circular imports
from typing import TYPE_CHECKING, Any

import jax.numpy as jnp
import pandas as pd

from jactus.core import ActusDateTime, ContractEvent, ContractState
from jactus.observers import ChildContractObserver, RiskFactorObserver

if TYPE_CHECKING:
    from jactus.contracts.base import BaseContract


@dataclass
class SimulationResult:
    """Results from a contract simulation.

    Contains complete history of events, states, and cashflows from a contract
    simulation. Provides utilities for analyzing and exporting results.

    Attributes:
        contract_id: Unique identifier for the simulated contract
        events: List of all contract events with payoffs
        states: List of all contract states through time
        initial_state: Contract state before first event
        final_state: Contract state after last event
        metadata: Additional simulation metadata (scenario info, timestamp, etc.)

    Example:
        >>> result = simulator.simulate_contract(contract)
        >>> print(f"Generated {len(result.events)} events")
        >>> print(f"Total cashflow: {result.total_cashflow()}")
        >>> df = result.to_dataframe()
    """

    contract_id: str
    events: list[ContractEvent]
    states: list[ContractState]
    initial_state: ContractState
    final_state: ContractState
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dataframe(self) -> pd.DataFrame:
        """Convert simulation results to pandas DataFrame.

        Returns:
            DataFrame with columns: time, event_type, payoff, currency,
            notional, interest, fees, etc.

        Example:
            >>> df = result.to_dataframe()
            >>> df.groupby('event_type')['payoff'].sum()
        """
        if not self.events:
            return pd.DataFrame()

        records = []
        for event in self.events:
            record = {
                "time": event.event_time.to_iso(),
                "event_type": str(event.event_type),
                "payoff": float(event.payoff),
                "currency": event.currency,
                "sequence": event.sequence,
            }

            # Add state information if available
            if event.state_post is not None:
                record["notional"] = float(event.state_post.nt)
                record["interest_accrued"] = float(event.state_post.ipac)
                record["fee_accrued"] = float(event.state_post.feac)
                record["nominal_interest_rate"] = float(event.state_post.ipnr)

            records.append(record)

        return pd.DataFrame(records)

    def get_cashflow_timeline(self) -> list[tuple[ActusDateTime, float, str]]:
        """Extract cashflow timeline as (time, amount, currency) tuples.

        Returns:
            List of (time, payoff, currency) tuples sorted by time

        Example:
            >>> timeline = result.get_cashflow_timeline()
            >>> for time, amount, currency in timeline:
            ...     print(f"{time}: {amount} {currency}")
        """
        return [(e.event_time, float(e.payoff), e.currency) for e in self.events]

    def filter_events(
        self,
        start: ActusDateTime | None = None,
        end: ActusDateTime | None = None,
        event_types: list[Any] | None = None,
    ) -> list[ContractEvent]:
        """Filter events by time range and/or event type.

        Args:
            start: Optional start time (inclusive)
            end: Optional end time (inclusive)
            event_types: Optional list of event types to include

        Returns:
            Filtered list of events

        Example:
            >>> from jactus.core import EventType
            >>> # Get only interest payment events in 2024
            >>> start = ActusDateTime(2024, 1, 1, 0, 0, 0)
            >>> end = ActusDateTime(2024, 12, 31, 23, 59, 59)
            >>> ip_events = result.filter_events(start, end, [EventType.IP])
        """
        filtered = self.events

        # Filter by time range
        if start is not None:
            filtered = [e for e in filtered if e.event_time >= start]
        if end is not None:
            filtered = [e for e in filtered if e.event_time <= end]

        # Filter by event type
        if event_types is not None:
            filtered = [e for e in filtered if e.event_type in event_types]

        return filtered

    def total_cashflow(self) -> float:
        """Calculate total cashflow across all events.

        Returns:
            Sum of all payoffs

        Note:
            Assumes all events have the same currency. For multi-currency
            contracts, use to_dataframe() and aggregate by currency.

        Example:
            >>> total = result.total_cashflow()
            >>> print(f"Net cashflow: {total}")
        """
        return sum(float(e.payoff) for e in self.events)

    def to_dict(self) -> dict[str, Any]:
        """Serialize result to dictionary.

        Returns:
            Dictionary representation suitable for JSON serialization

        Example:
            >>> import json
            >>> result_dict = result.to_dict()
            >>> json.dumps(result_dict, indent=2)
        """
        return {
            "contract_id": self.contract_id,
            "num_events": len(self.events),
            "num_states": len(self.states),
            "initial_state": {
                "sd": self.initial_state.sd.to_iso(),
                "nt": float(self.initial_state.nt),
                "ipnr": float(self.initial_state.ipnr),
            },
            "final_state": {
                "sd": self.final_state.sd.to_iso(),
                "nt": float(self.final_state.nt),
                "ipnr": float(self.final_state.ipnr),
            },
            "total_cashflow": self.total_cashflow(),
            "metadata": self.metadata,
        }


class ContractSimulator:
    """High-level contract simulation engine.

    Provides methods for simulating ACTUS contracts under different scenarios
    and aggregating results. Supports single contract simulation, scenario
    analysis, and batch processing.

    Attributes:
        default_risk_factor_observer: Default observer for market data
        default_child_contract_observer: Default observer for child contracts

    Example:
        >>> # Create simulator with default observers
        >>> from jactus.observers import ConstantRiskFactorObserver
        >>> rf_observer = ConstantRiskFactorObserver(default_value=0.05)
        >>> simulator = ContractSimulator(default_risk_factor_observer=rf_observer)
        >>>
        >>> # Simulate contract
        >>> result = simulator.simulate_contract(contract)
        >>> print(result.to_dataframe())
    """

    def __init__(
        self,
        default_risk_factor_observer: RiskFactorObserver | None = None,
        default_child_contract_observer: ChildContractObserver | None = None,
    ):
        """Initialize the contract simulator.

        Args:
            default_risk_factor_observer: Optional default observer for market data
            default_child_contract_observer: Optional default observer for child contracts
        """
        self.default_risk_factor_observer = default_risk_factor_observer
        self.default_child_contract_observer = default_child_contract_observer

    def simulate_contract(
        self,
        contract: "BaseContract",
        risk_factor_observer: RiskFactorObserver | None = None,
        child_contract_observer: ChildContractObserver | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> SimulationResult:
        """Simulate a single contract through its lifecycle.

        Args:
            contract: Contract to simulate
            risk_factor_observer: Optional observer for market data (overrides default)
            child_contract_observer: Optional observer for child contracts (overrides default)
            metadata: Optional metadata to attach to result

        Returns:
            SimulationResult containing complete simulation history

        Example:
            >>> result = simulator.simulate_contract(contract)
            >>> df = result.to_dataframe()
            >>> print(f"Generated {len(result.events)} events")
        """
        # Use provided observers or fall back to defaults
        rf_obs = risk_factor_observer or self.default_risk_factor_observer
        child_obs = child_contract_observer or self.default_child_contract_observer

        # Run simulation using contract's simulate method
        history = contract.simulate(
            risk_factor_observer=rf_obs,
            child_contract_observer=child_obs,
        )

        # Convert to SimulationResult
        return SimulationResult(
            contract_id=contract.attributes.contract_id,
            events=history.events,
            states=history.states,
            initial_state=history.initial_state,
            final_state=history.final_state,
            metadata=metadata or {},
        )

    def simulate_scenario(
        self,
        contract: "BaseContract",
        scenario_name: str,
        risk_factor_observer: RiskFactorObserver,
        child_contract_observer: ChildContractObserver | None = None,
    ) -> SimulationResult:
        """Simulate contract under a specific scenario.

        Args:
            contract: Contract to simulate
            scenario_name: Name of the scenario (stored in metadata)
            risk_factor_observer: Scenario-specific risk factor observer
            child_contract_observer: Optional child contract observer

        Returns:
            SimulationResult with scenario metadata

        Example:
            >>> from jactus.observers import ConstantRiskFactorObserver
            >>> # Base case
            >>> base_rf = ConstantRiskFactorObserver(default_value=0.05)
            >>> base_result = simulator.simulate_scenario(
            ...     contract, "base_case", base_rf
            ... )
            >>> # Stress scenario
            >>> stress_rf = ConstantRiskFactorObserver(default_value=0.10)
            >>> stress_result = simulator.simulate_scenario(
            ...     contract, "stress_case", stress_rf
            ... )
        """
        metadata = {"scenario": scenario_name}
        return self.simulate_contract(
            contract=contract,
            risk_factor_observer=risk_factor_observer,
            child_contract_observer=child_contract_observer,
            metadata=metadata,
        )

    def simulate_multiple_scenarios(
        self,
        contract: "BaseContract",
        scenarios: dict[str, RiskFactorObserver],
        child_contract_observer: ChildContractObserver | None = None,
    ) -> dict[str, SimulationResult]:
        """Simulate contract under multiple scenarios.

        Args:
            contract: Contract to simulate
            scenarios: Dictionary mapping scenario names to risk factor observers
            child_contract_observer: Optional child contract observer (same for all scenarios)

        Returns:
            Dictionary mapping scenario names to SimulationResults

        Example:
            >>> scenarios = {
            ...     "base": ConstantRiskFactorObserver(0.05),
            ...     "bull": ConstantRiskFactorObserver(0.03),
            ...     "bear": ConstantRiskFactorObserver(0.08),
            ... }
            >>> results = simulator.simulate_multiple_scenarios(contract, scenarios)
            >>> for name, result in results.items():
            ...     print(f"{name}: {result.total_cashflow()}")
        """
        results = {}
        for scenario_name, rf_observer in scenarios.items():
            results[scenario_name] = self.simulate_scenario(
                contract=contract,
                scenario_name=scenario_name,
                risk_factor_observer=rf_observer,
                child_contract_observer=child_contract_observer,
            )
        return results


def simulate_contracts(
    contracts: list["BaseContract"],
    risk_factor_observer: RiskFactorObserver,
    child_contract_observer: ChildContractObserver | None = None,
) -> list[SimulationResult]:
    """Simulate multiple contracts sequentially.

    Note: This is a simple sequential implementation. For parallel simulation
    with JAX vmap, contracts need to be pytree-compatible and use pure functions.
    See the advanced batch simulation utilities for vectorized processing.

    Args:
        contracts: List of contracts to simulate
        risk_factor_observer: Observer for market data (shared across contracts)
        child_contract_observer: Optional observer for child contracts

    Returns:
        List of SimulationResults, one per contract

    Example:
        >>> from jactus.observers import ConstantRiskFactorObserver
        >>> rf_obs = ConstantRiskFactorObserver(0.05)
        >>> results = simulate_contracts(contracts, rf_obs)
        >>> total = sum(r.total_cashflow() for r in results)
    """
    simulator = ContractSimulator(
        default_risk_factor_observer=risk_factor_observer,
        default_child_contract_observer=child_contract_observer,
    )

    results = []
    for contract in contracts:
        result = simulator.simulate_contract(contract)
        results.append(result)

    return results


def create_cashflow_matrix(
    results: list[SimulationResult],
    time_points: list[ActusDateTime],
) -> jnp.ndarray:
    """Create cashflow matrix from multiple simulation results.

    Creates a 2D matrix where rows are contracts and columns are time points.
    Each cell contains the total cashflow for that contract at that time point.

    Args:
        results: List of simulation results
        time_points: List of time points to sample (columns)

    Returns:
        JAX array of shape (num_contracts, num_time_points) with cashflows

    Example:
        >>> # Create quarterly time points for 2024
        >>> time_points = [
        ...     ActusDateTime(2024, 3, 31, 0, 0, 0),
        ...     ActusDateTime(2024, 6, 30, 0, 0, 0),
        ...     ActusDateTime(2024, 9, 30, 0, 0, 0),
        ...     ActusDateTime(2024, 12, 31, 0, 0, 0),
        ... ]
        >>> matrix = create_cashflow_matrix(results, time_points)
        >>> # Sum cashflows per quarter across all contracts
        >>> quarterly_totals = matrix.sum(axis=0)
    """
    num_contracts = len(results)
    num_time_points = len(time_points)

    # Initialize matrix
    cashflow_matrix = jnp.zeros((num_contracts, num_time_points))

    # Fill matrix
    for i, result in enumerate(results):
        for j, time_point in enumerate(time_points):
            # Sum all cashflows at this time point
            total_cf = sum(float(e.payoff) for e in result.events if e.event_time == time_point)
            cashflow_matrix = cashflow_matrix.at[i, j].set(total_cf)

    return cashflow_matrix
