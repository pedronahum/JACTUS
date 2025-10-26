"""Unit tests for contract simulation engine.

Tests the ContractSimulator class and SimulationResult dataclass for running
ACTUS contracts through their lifecycles.
"""

from abc import ABC
from typing import Any

import jax.numpy as jnp
import pandas as pd

from jactus.core import (
    ActusDateTime,
    ContractAttributes,
    ContractEvent,
    ContractRole,
    ContractState,
    ContractType,
    EventSchedule,
    EventType,
)
from jactus.engine import (
    ContractSimulator,
    SimulationResult,
    create_cashflow_matrix,
    simulate_contracts,
)
from jactus.functions import PayoffFunction, StateTransitionFunction
from jactus.observers import ConstantRiskFactorObserver, RiskFactorObserver

# Import BaseContract for testing
try:
    from jactus.contracts.base import BaseContract
except ImportError:
    # Define minimal version if not available
    import flax.nnx as nnx

    class BaseContract(nnx.Module, ABC):
        """Minimal BaseContract for testing."""

        def __init__(
            self,
            attributes: ContractAttributes,
            risk_factor_observer: RiskFactorObserver,
            child_contract_observer=None,
            *,
            rngs=None,
        ):
            super().__init__()
            self.attributes = attributes
            self.risk_factor_observer = risk_factor_observer
            self.child_contract_observer = child_contract_observer
            self.rngs = rngs if rngs is not None else nnx.Rngs(0)


# ============================================================================
# Mock Implementations
# ============================================================================


class MockPayoffFunction:
    """Simple payoff function that returns fixed amounts."""

    def __init__(self, amount: float = 100.0):
        self.amount = amount

    def __call__(
        self,
        event_type: Any,
        state: ContractState,
        attributes: ContractAttributes,
        time: ActusDateTime,
        risk_factor_observer: RiskFactorObserver,
    ) -> jnp.ndarray:
        """Return fixed payoff amount."""
        return jnp.array(self.amount, dtype=jnp.float32)


class MockStateTransitionFunction:
    """Simple STF that updates status date."""

    def __call__(
        self,
        event_type: Any,
        state_pre: ContractState,
        attributes: ContractAttributes,
        time: ActusDateTime,
        risk_factor_observer: RiskFactorObserver,
    ) -> ContractState:
        """Update status date to event time."""
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


class MockContract(BaseContract):
    """Mock contract for testing simulation."""

    def __init__(
        self,
        attributes: ContractAttributes,
        risk_factor_observer: RiskFactorObserver,
        num_events: int = 3,
        payoff_amount: float = 100.0,
        **kwargs,
    ):
        super().__init__(
            attributes=attributes,
            risk_factor_observer=risk_factor_observer,
            **kwargs,
        )
        self.num_events = num_events
        self.payoff_amount = payoff_amount
        self._pof = MockPayoffFunction(payoff_amount)
        self._stf = MockStateTransitionFunction()

    def generate_event_schedule(self) -> EventSchedule:
        """Generate simple monthly event schedule."""
        events = []
        for i in range(self.num_events):
            event_time = self.attributes.status_date.add_period(f"{i+1}M")
            event = ContractEvent(
                event_type=EventType.IP,
                event_time=event_time,
                payoff=jnp.array(0.0),
                currency=self.attributes.currency or "USD",
                state_pre=None,
                state_post=None,
                sequence=i,
            )
            events.append(event)

        return EventSchedule(
            events=tuple(events),
            contract_id=self.attributes.contract_id,
        )

    def initialize_state(self) -> ContractState:
        """Create initial state."""
        return ContractState(
            sd=self.attributes.status_date,
            tmd=self.attributes.status_date,
            nt=self.attributes.notional_principal or jnp.array(10000.0),
            ipnr=self.attributes.nominal_interest_rate or jnp.array(0.05),
            ipac=jnp.array(0.0),
            feac=jnp.array(0.0),
            nsc=jnp.array(1.0),
            isc=jnp.array(1.0),
        )

    def get_payoff_function(self, event_type: Any) -> PayoffFunction:
        """Return mock payoff function."""
        return self._pof

    def get_state_transition_function(self, event_type: Any) -> StateTransitionFunction:
        """Return mock STF."""
        return self._stf


# ============================================================================
# Test SimulationResult
# ============================================================================


class TestSimulationResult:
    """Test SimulationResult dataclass methods."""

    def test_initialization(self):
        """Test SimulationResult can be created."""
        initial = ContractState(
            sd=ActusDateTime(2024, 1, 1, 0, 0, 0),
            tmd=ActusDateTime(2024, 1, 1, 0, 0, 0),
            nt=jnp.array(10000.0),
            ipnr=jnp.array(0.05),
            ipac=jnp.array(0.0),
            feac=jnp.array(0.0),
            nsc=jnp.array(1.0),
            isc=jnp.array(1.0),
        )

        result = SimulationResult(
            contract_id="TEST001",
            events=[],
            states=[],
            initial_state=initial,
            final_state=initial,
        )

        assert result.contract_id == "TEST001"
        assert len(result.events) == 0
        assert result.initial_state == initial

    def test_to_dataframe_empty(self):
        """Test to_dataframe with no events."""
        initial = ContractState(
            sd=ActusDateTime(2024, 1, 1, 0, 0, 0),
            tmd=ActusDateTime(2024, 1, 1, 0, 0, 0),
            nt=jnp.array(10000.0),
            ipnr=jnp.array(0.05),
            ipac=jnp.array(0.0),
            feac=jnp.array(0.0),
            nsc=jnp.array(1.0),
            isc=jnp.array(1.0),
        )

        result = SimulationResult(
            contract_id="TEST001",
            events=[],
            states=[],
            initial_state=initial,
            final_state=initial,
        )

        df = result.to_dataframe()
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0

    def test_to_dataframe_with_events(self):
        """Test to_dataframe with events."""
        state = ContractState(
            sd=ActusDateTime(2024, 1, 1, 0, 0, 0),
            tmd=ActusDateTime(2024, 1, 1, 0, 0, 0),
            nt=jnp.array(10000.0),
            ipnr=jnp.array(0.05),
            ipac=jnp.array(50.0),
            feac=jnp.array(10.0),
            nsc=jnp.array(1.0),
            isc=jnp.array(1.0),
        )

        events = [
            ContractEvent(
                event_type=EventType.IP,
                event_time=ActusDateTime(2024, 2, 1, 0, 0, 0),
                payoff=jnp.array(100.0),
                currency="USD",
                state_pre=state,
                state_post=state,
                sequence=0,
            ),
            ContractEvent(
                event_type=EventType.IP,
                event_time=ActusDateTime(2024, 3, 1, 0, 0, 0),
                payoff=jnp.array(100.0),
                currency="USD",
                state_pre=state,
                state_post=state,
                sequence=1,
            ),
        ]

        result = SimulationResult(
            contract_id="TEST001",
            events=events,
            states=[state, state],
            initial_state=state,
            final_state=state,
        )

        df = result.to_dataframe()
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2
        assert "time" in df.columns
        assert "event_type" in df.columns
        assert "payoff" in df.columns
        assert "notional" in df.columns
        assert df["payoff"].sum() == 200.0

    def test_get_cashflow_timeline(self):
        """Test get_cashflow_timeline extraction."""
        state = ContractState(
            sd=ActusDateTime(2024, 1, 1, 0, 0, 0),
            tmd=ActusDateTime(2024, 1, 1, 0, 0, 0),
            nt=jnp.array(10000.0),
            ipnr=jnp.array(0.05),
            ipac=jnp.array(0.0),
            feac=jnp.array(0.0),
            nsc=jnp.array(1.0),
            isc=jnp.array(1.0),
        )

        events = [
            ContractEvent(
                event_type=EventType.IP,
                event_time=ActusDateTime(2024, 2, 1, 0, 0, 0),
                payoff=jnp.array(100.0),
                currency="USD",
                state_pre=state,
                state_post=state,
                sequence=0,
            ),
        ]

        result = SimulationResult(
            contract_id="TEST001",
            events=events,
            states=[state],
            initial_state=state,
            final_state=state,
        )

        timeline = result.get_cashflow_timeline()
        assert len(timeline) == 1
        time, amount, currency = timeline[0]
        assert time == ActusDateTime(2024, 2, 1, 0, 0, 0)
        assert amount == 100.0
        assert currency == "USD"

    def test_filter_events_by_time(self):
        """Test filtering events by time range."""
        state = ContractState(
            sd=ActusDateTime(2024, 1, 1, 0, 0, 0),
            tmd=ActusDateTime(2024, 1, 1, 0, 0, 0),
            nt=jnp.array(10000.0),
            ipnr=jnp.array(0.05),
            ipac=jnp.array(0.0),
            feac=jnp.array(0.0),
            nsc=jnp.array(1.0),
            isc=jnp.array(1.0),
        )

        events = [
            ContractEvent(
                event_type=EventType.IP,
                event_time=ActusDateTime(2024, i, 1, 0, 0, 0),
                payoff=jnp.array(100.0),
                currency="USD",
                state_pre=state,
                state_post=state,
                sequence=i - 1,
            )
            for i in range(1, 5)
        ]

        result = SimulationResult(
            contract_id="TEST001",
            events=events,
            states=[state] * 4,
            initial_state=state,
            final_state=state,
        )

        # Filter Feb to Mar
        filtered = result.filter_events(
            start=ActusDateTime(2024, 2, 1, 0, 0, 0),
            end=ActusDateTime(2024, 3, 31, 0, 0, 0),
        )
        assert len(filtered) == 2

    def test_filter_events_by_type(self):
        """Test filtering events by event type."""
        state = ContractState(
            sd=ActusDateTime(2024, 1, 1, 0, 0, 0),
            tmd=ActusDateTime(2024, 1, 1, 0, 0, 0),
            nt=jnp.array(10000.0),
            ipnr=jnp.array(0.05),
            ipac=jnp.array(0.0),
            feac=jnp.array(0.0),
            nsc=jnp.array(1.0),
            isc=jnp.array(1.0),
        )

        events = [
            ContractEvent(
                event_type=EventType.IP,
                event_time=ActusDateTime(2024, 1, 1, 0, 0, 0),
                payoff=jnp.array(100.0),
                currency="USD",
                state_pre=state,
                state_post=state,
                sequence=0,
            ),
            ContractEvent(
                event_type=EventType.PR,
                event_time=ActusDateTime(2024, 2, 1, 0, 0, 0),
                payoff=jnp.array(1000.0),
                currency="USD",
                state_pre=state,
                state_post=state,
                sequence=1,
            ),
        ]

        result = SimulationResult(
            contract_id="TEST001",
            events=events,
            states=[state, state],
            initial_state=state,
            final_state=state,
        )

        # Filter only IP events
        filtered = result.filter_events(event_types=[EventType.IP])
        assert len(filtered) == 1
        assert filtered[0].event_type == EventType.IP

    def test_total_cashflow(self):
        """Test total cashflow calculation."""
        state = ContractState(
            sd=ActusDateTime(2024, 1, 1, 0, 0, 0),
            tmd=ActusDateTime(2024, 1, 1, 0, 0, 0),
            nt=jnp.array(10000.0),
            ipnr=jnp.array(0.05),
            ipac=jnp.array(0.0),
            feac=jnp.array(0.0),
            nsc=jnp.array(1.0),
            isc=jnp.array(1.0),
        )

        events = [
            ContractEvent(
                event_type=EventType.IP,
                event_time=ActusDateTime(2024, i, 1, 0, 0, 0),
                payoff=jnp.array(100.0),
                currency="USD",
                state_pre=state,
                state_post=state,
                sequence=i - 1,
            )
            for i in range(1, 4)
        ]

        result = SimulationResult(
            contract_id="TEST001",
            events=events,
            states=[state] * 3,
            initial_state=state,
            final_state=state,
        )

        total = result.total_cashflow()
        assert total == 300.0

    def test_to_dict(self):
        """Test serialization to dictionary."""
        state = ContractState(
            sd=ActusDateTime(2024, 1, 1, 0, 0, 0),
            tmd=ActusDateTime(2024, 1, 1, 0, 0, 0),
            nt=jnp.array(10000.0),
            ipnr=jnp.array(0.05),
            ipac=jnp.array(0.0),
            feac=jnp.array(0.0),
            nsc=jnp.array(1.0),
            isc=jnp.array(1.0),
        )

        result = SimulationResult(
            contract_id="TEST001",
            events=[],
            states=[],
            initial_state=state,
            final_state=state,
            metadata={"scenario": "base"},
        )

        d = result.to_dict()
        assert d["contract_id"] == "TEST001"
        assert d["num_events"] == 0
        assert d["metadata"]["scenario"] == "base"
        assert "initial_state" in d
        assert "final_state" in d


# ============================================================================
# Test ContractSimulator
# ============================================================================


class TestContractSimulator:
    """Test ContractSimulator class."""

    def test_initialization(self):
        """Test ContractSimulator can be created."""
        simulator = ContractSimulator()
        assert simulator.default_risk_factor_observer is None
        assert simulator.default_child_contract_observer is None

    def test_initialization_with_defaults(self):
        """Test ContractSimulator with default observers."""
        rf_obs = ConstantRiskFactorObserver(constant_value=0.05)
        simulator = ContractSimulator(default_risk_factor_observer=rf_obs)
        assert simulator.default_risk_factor_observer == rf_obs

    def test_simulate_contract(self):
        """Test simulating a single contract."""
        attrs = ContractAttributes(
            contract_id="TEST001",
            contract_type=ContractType.PAM,
            contract_role=ContractRole.RPA,
            status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
            currency="USD",
        )

        rf_obs = ConstantRiskFactorObserver(constant_value=0.05)
        contract = MockContract(attributes=attrs, risk_factor_observer=rf_obs, num_events=3)

        simulator = ContractSimulator()
        result = simulator.simulate_contract(contract, risk_factor_observer=rf_obs)

        assert isinstance(result, SimulationResult)
        assert result.contract_id == "TEST001"
        assert len(result.events) == 3
        assert len(result.states) == 3

    def test_simulate_contract_with_default_observer(self):
        """Test simulation using default observer."""
        attrs = ContractAttributes(
            contract_id="TEST001",
            contract_type=ContractType.PAM,
            contract_role=ContractRole.RPA,
            status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
            currency="USD",
        )

        rf_obs = ConstantRiskFactorObserver(constant_value=0.05)
        contract = MockContract(attributes=attrs, risk_factor_observer=rf_obs)

        simulator = ContractSimulator(default_risk_factor_observer=rf_obs)
        result = simulator.simulate_contract(contract)

        assert isinstance(result, SimulationResult)
        assert len(result.events) == 3

    def test_simulate_scenario(self):
        """Test scenario simulation with metadata."""
        attrs = ContractAttributes(
            contract_id="TEST001",
            contract_type=ContractType.PAM,
            contract_role=ContractRole.RPA,
            status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
            currency="USD",
        )

        rf_obs = ConstantRiskFactorObserver(constant_value=0.05)
        contract = MockContract(attributes=attrs, risk_factor_observer=rf_obs)

        simulator = ContractSimulator()
        result = simulator.simulate_scenario(
            contract=contract, scenario_name="base_case", risk_factor_observer=rf_obs
        )

        assert result.metadata["scenario"] == "base_case"

    def test_simulate_multiple_scenarios(self):
        """Test simulating multiple scenarios."""
        attrs = ContractAttributes(
            contract_id="TEST001",
            contract_type=ContractType.PAM,
            contract_role=ContractRole.RPA,
            status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
            currency="USD",
        )

        base_rf = ConstantRiskFactorObserver(constant_value=0.05)
        contract = MockContract(attributes=attrs, risk_factor_observer=base_rf, payoff_amount=100.0)

        scenarios = {
            "base": ConstantRiskFactorObserver(constant_value=0.05),
            "stress": ConstantRiskFactorObserver(constant_value=0.10),
        }

        simulator = ContractSimulator()
        results = simulator.simulate_multiple_scenarios(contract, scenarios)

        assert len(results) == 2
        assert "base" in results
        assert "stress" in results
        assert results["base"].metadata["scenario"] == "base"
        assert results["stress"].metadata["scenario"] == "stress"


# ============================================================================
# Test Batch Functions
# ============================================================================


class TestBatchSimulation:
    """Test batch simulation functions."""

    def test_simulate_contracts(self):
        """Test simulating multiple contracts."""
        contracts = []
        rf_obs = ConstantRiskFactorObserver(constant_value=0.05)

        for i in range(3):
            attrs = ContractAttributes(
                contract_id=f"TEST{i:03d}",
                contract_type=ContractType.PAM,
                contract_role=ContractRole.RPA,
                status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
                currency="USD",
            )
            contract = MockContract(attributes=attrs, risk_factor_observer=rf_obs, num_events=2)
            contracts.append(contract)

        results = simulate_contracts(contracts, rf_obs)

        assert len(results) == 3
        assert all(isinstance(r, SimulationResult) for r in results)
        assert all(len(r.events) == 2 for r in results)


class TestCashflowMatrix:
    """Test cashflow matrix creation."""

    def test_create_cashflow_matrix(self):
        """Test creating cashflow matrix."""
        # Create mock results
        state = ContractState(
            sd=ActusDateTime(2024, 1, 1, 0, 0, 0),
            tmd=ActusDateTime(2024, 1, 1, 0, 0, 0),
            nt=jnp.array(10000.0),
            ipnr=jnp.array(0.05),
            ipac=jnp.array(0.0),
            feac=jnp.array(0.0),
            nsc=jnp.array(1.0),
            isc=jnp.array(1.0),
        )

        time_points = [
            ActusDateTime(2024, 1, 31, 0, 0, 0),
            ActusDateTime(2024, 2, 29, 0, 0, 0),
            ActusDateTime(2024, 3, 31, 0, 0, 0),
        ]

        # Create 2 contracts with events at different times
        results = []
        for contract_num in range(2):
            events = [
                ContractEvent(
                    event_type=EventType.IP,
                    event_time=time_points[i],
                    payoff=jnp.array(100.0 * (contract_num + 1)),
                    currency="USD",
                    state_pre=state,
                    state_post=state,
                    sequence=i,
                )
                for i in range(3)
            ]

            result = SimulationResult(
                contract_id=f"TEST{contract_num}",
                events=events,
                states=[state] * 3,
                initial_state=state,
                final_state=state,
            )
            results.append(result)

        # Create matrix
        matrix = create_cashflow_matrix(results, time_points)

        assert matrix.shape == (2, 3)
        # First contract has 100 at each time point
        assert float(matrix[0, 0]) == 100.0
        # Second contract has 200 at each time point
        assert float(matrix[1, 0]) == 200.0

    def test_create_cashflow_matrix_empty(self):
        """Test creating matrix with no cashflows at time points."""
        state = ContractState(
            sd=ActusDateTime(2024, 1, 1, 0, 0, 0),
            tmd=ActusDateTime(2024, 1, 1, 0, 0, 0),
            nt=jnp.array(10000.0),
            ipnr=jnp.array(0.05),
            ipac=jnp.array(0.0),
            feac=jnp.array(0.0),
            nsc=jnp.array(1.0),
            isc=jnp.array(1.0),
        )

        result = SimulationResult(
            contract_id="TEST001",
            events=[],
            states=[],
            initial_state=state,
            final_state=state,
        )

        time_points = [
            ActusDateTime(2024, 1, 31, 0, 0, 0),
            ActusDateTime(2024, 2, 29, 0, 0, 0),
        ]

        matrix = create_cashflow_matrix([result], time_points)

        assert matrix.shape == (1, 2)
        assert float(matrix[0, 0]) == 0.0
        assert float(matrix[0, 1]) == 0.0
