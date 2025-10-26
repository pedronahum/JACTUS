"""Unit tests for base contract class.

T2.5: Base Contract Tests

Tests for:
- BaseContract abstract methods
- Flax NNX integration
- Contract simulation
- NPV calculation
- Event caching
- Helper functions
"""

import jax.numpy as jnp
import pytest
from flax import nnx

from jactus.contracts import (
    BaseContract,
    SimulationHistory,
    merge_scheduled_and_observed_events,
    sort_events_by_sequence,
)
from jactus.core import (
    ActusDateTime,
    ContractAttributes,
    ContractEvent,
    ContractState,
    EventSchedule,
)
from jactus.core.types import ContractRole, ContractType, EventType
from jactus.functions import BasePayoffFunction, BaseStateTransitionFunction
from jactus.observers import ConstantRiskFactorObserver

# ============================================================================
# Mock implementations for testing
# ============================================================================


class MockPayoffFunction(BasePayoffFunction):
    """Mock payoff function that returns constant payoff."""

    def __init__(self, payoff_amount: float = 100.0, contract_role=None, currency="USD"):
        super().__init__(
            contract_role=contract_role or ContractRole.RPA,
            currency=currency,
            settlement_currency=None,
        )
        self.payoff_amount = payoff_amount

    def calculate_payoff(
        self,
        event_type,
        state,
        attributes,
        time,
        risk_factor_observer,
    ) -> jnp.ndarray:
        """Return constant payoff."""
        return jnp.array(self.payoff_amount, dtype=jnp.float32)


class MockStateTransitionFunction(BaseStateTransitionFunction):
    """Mock STF that updates status date only."""

    def transition_state(
        self,
        event_type,
        state_pre,
        attributes,
        time,
        risk_factor_observer,
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
    """Concrete implementation of BaseContract for testing."""

    def __init__(self, attributes, risk_factor_observer, num_events=3, **kwargs):
        super().__init__(attributes, risk_factor_observer, **kwargs)
        self.num_events = num_events

    def generate_event_schedule(self) -> EventSchedule:
        """Generate simple event schedule."""
        events = []
        start = self.attributes.status_date

        for i in range(self.num_events):
            # Create event every month
            event_time = start.add_period(f"{i}M")
            events.append(
                ContractEvent(
                    event_type=EventType.IP if i > 0 else EventType.IED,
                    event_time=event_time,
                    payoff=jnp.array(0.0),
                    currency="USD",
                )
            )

        return EventSchedule(
            events=tuple(events),
            contract_id=self.attributes.contract_id,
        )

    def initialize_state(self) -> ContractState:
        """Initialize mock state."""
        return ContractState(
            sd=self.attributes.status_date,
            tmd=self.attributes.maturity_date or self.attributes.status_date,
            nt=jnp.array(self.attributes.notional_principal or 100000.0),
            ipnr=jnp.array(self.attributes.nominal_interest_rate or 0.05),
            ipac=jnp.array(0.0),
            feac=jnp.array(0.0),
            nsc=jnp.array(1.0),
            isc=jnp.array(1.0),
        )

    def get_payoff_function(self, event_type):
        """Return mock payoff function."""
        if event_type == EventType.IED:
            return MockPayoffFunction(0.0)  # No payoff for IED
        return MockPayoffFunction(100.0)  # 100 for other events

    def get_state_transition_function(self, event_type):
        """Return mock state transition function."""
        return MockStateTransitionFunction()


# ============================================================================
# Test BaseContract abstract class
# ============================================================================


class TestBaseContractAbstract:
    """Test that BaseContract enforces abstract methods."""

    def test_cannot_instantiate_base_contract(self):
        """Test that BaseContract cannot be instantiated directly."""
        attrs = ContractAttributes(
            contract_id="TEST",
            contract_type=ContractType.PAM,
            contract_role=ContractRole.RPA,
            status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
        )
        risk_obs = ConstantRiskFactorObserver(1.0)

        with pytest.raises(TypeError):
            BaseContract(attrs, risk_obs)  # type: ignore

    def test_subclass_must_implement_generate_event_schedule(self):
        """Test that subclass must implement generate_event_schedule."""

        class IncompleteContract(BaseContract):
            def initialize_state(self):
                return ContractState(
                    sd=ActusDateTime(2024, 1, 1, 0, 0, 0),
                    tmd=ActusDateTime(2025, 1, 1, 0, 0, 0),
                    nt=jnp.array(100000.0),
                    ipnr=jnp.array(0.05),
                    ipac=jnp.array(0.0),
                    feac=jnp.array(0.0),
                    nsc=jnp.array(1.0),
                    isc=jnp.array(1.0),
                )

            def get_payoff_function(self, event_type):
                return MockPayoffFunction()

            def get_state_transition_function(self, event_type):
                return MockStateTransitionFunction()

        attrs = ContractAttributes(
            contract_id="TEST",
            contract_type=ContractType.PAM,
            contract_role=ContractRole.RPA,
            status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
        )
        risk_obs = ConstantRiskFactorObserver(1.0)

        with pytest.raises(TypeError):
            IncompleteContract(attrs, risk_obs)  # type: ignore


class TestFlaxNNXIntegration:
    """Test Flax NNX module integration."""

    def test_base_contract_is_nnx_module(self):
        """Test that BaseContract is a Flax NNX Module."""
        attrs = ContractAttributes(
            contract_id="TEST",
            contract_type=ContractType.PAM,
            contract_role=ContractRole.RPA,
            status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
        )
        risk_obs = ConstantRiskFactorObserver(1.0)
        contract = MockContract(attrs, risk_obs)

        assert isinstance(contract, nnx.Module)

    def test_contract_has_rngs(self):
        """Test that contract has RNG state."""
        attrs = ContractAttributes(
            contract_id="TEST",
            contract_type=ContractType.PAM,
            contract_role=ContractRole.RPA,
            status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
        )
        risk_obs = ConstantRiskFactorObserver(1.0)
        contract = MockContract(attrs, risk_obs)

        assert hasattr(contract, "rngs")
        assert isinstance(contract.rngs, nnx.Rngs)

    def test_contract_with_custom_rngs(self):
        """Test contract initialization with custom RNGs."""
        attrs = ContractAttributes(
            contract_id="TEST",
            contract_type=ContractType.PAM,
            contract_role=ContractRole.RPA,
            status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
        )
        risk_obs = ConstantRiskFactorObserver(1.0)
        rngs = nnx.Rngs(42)
        contract = MockContract(attrs, risk_obs, rngs=rngs)

        assert contract.rngs is rngs


class TestContractInitialization:
    """Test contract initialization."""

    def test_init_with_required_args(self):
        """Test initialization with required arguments."""
        attrs = ContractAttributes(
            contract_id="TEST",
            contract_type=ContractType.PAM,
            contract_role=ContractRole.RPA,
            status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
        )
        risk_obs = ConstantRiskFactorObserver(1.0)
        contract = MockContract(attrs, risk_obs)

        assert contract.attributes == attrs
        assert contract.risk_factor_observer == risk_obs
        assert contract.child_contract_observer is None
        assert contract._event_cache is None

    def test_init_with_child_observer(self):
        """Test initialization with child contract observer."""
        from jactus.observers import MockChildContractObserver

        attrs = ContractAttributes(
            contract_id="TEST",
            contract_type=ContractType.PAM,
            contract_role=ContractRole.RPA,
            status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
        )
        risk_obs = ConstantRiskFactorObserver(1.0)
        child_obs = MockChildContractObserver()
        contract = MockContract(attrs, risk_obs, child_contract_observer=child_obs)

        assert contract.child_contract_observer == child_obs


class TestLifetimeManagement:
    """Test contract lifetime methods."""

    def test_get_lifetime_with_maturity(self):
        """Test get_lifetime when maturity date is set."""
        start = ActusDateTime(2024, 1, 1, 0, 0, 0)
        end = ActusDateTime(2025, 1, 1, 0, 0, 0)
        attrs = ContractAttributes(
            contract_id="TEST",
            contract_type=ContractType.PAM,
            contract_role=ContractRole.RPA,
            status_date=start,
            maturity_date=end,
        )
        risk_obs = ConstantRiskFactorObserver(1.0)
        contract = MockContract(attrs, risk_obs)

        lifetime_start, lifetime_end = contract.get_lifetime()
        assert lifetime_start == start
        assert lifetime_end == end

    def test_get_lifetime_without_maturity(self):
        """Test get_lifetime when maturity date is None."""
        start = ActusDateTime(2024, 1, 1, 0, 0, 0)
        attrs = ContractAttributes(
            contract_id="TEST",
            contract_type=ContractType.PAM,
            contract_role=ContractRole.RPA,
            status_date=start,
        )
        risk_obs = ConstantRiskFactorObserver(1.0)
        contract = MockContract(attrs, risk_obs)

        lifetime_start, lifetime_end = contract.get_lifetime()
        assert lifetime_start == start
        assert lifetime_end == start  # Falls back to status_date

    def test_is_maturity_contract_true(self):
        """Test is_maturity_contract when maturity is set."""
        attrs = ContractAttributes(
            contract_id="TEST",
            contract_type=ContractType.PAM,
            contract_role=ContractRole.RPA,
            status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
            maturity_date=ActusDateTime(2025, 1, 1, 0, 0, 0),
        )
        risk_obs = ConstantRiskFactorObserver(1.0)
        contract = MockContract(attrs, risk_obs)

        assert contract.is_maturity_contract() is True

    def test_is_maturity_contract_false(self):
        """Test is_maturity_contract when maturity is None."""
        attrs = ContractAttributes(
            contract_id="TEST",
            contract_type=ContractType.PAM,
            contract_role=ContractRole.RPA,
            status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
        )
        risk_obs = ConstantRiskFactorObserver(1.0)
        contract = MockContract(attrs, risk_obs)

        assert contract.is_maturity_contract() is False


class TestEventScheduleCaching:
    """Test event schedule generation and caching."""

    def test_get_events_generates_schedule(self):
        """Test that get_events generates event schedule."""
        attrs = ContractAttributes(
            contract_id="TEST",
            contract_type=ContractType.PAM,
            contract_role=ContractRole.RPA,
            status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
        )
        risk_obs = ConstantRiskFactorObserver(1.0)
        contract = MockContract(attrs, risk_obs, num_events=3)

        events = contract.get_events()
        assert isinstance(events, EventSchedule)
        assert len(events.events) == 3

    def test_get_events_caches_schedule(self):
        """Test that get_events caches the schedule."""
        attrs = ContractAttributes(
            contract_id="TEST",
            contract_type=ContractType.PAM,
            contract_role=ContractRole.RPA,
            status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
        )
        risk_obs = ConstantRiskFactorObserver(1.0)
        contract = MockContract(attrs, risk_obs)

        events1 = contract.get_events()
        events2 = contract.get_events()

        # Should return same cached object
        assert events1 is events2

    def test_get_events_force_regenerate(self):
        """Test that force_regenerate bypasses cache."""
        attrs = ContractAttributes(
            contract_id="TEST",
            contract_type=ContractType.PAM,
            contract_role=ContractRole.RPA,
            status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
        )
        risk_obs = ConstantRiskFactorObserver(1.0)
        contract = MockContract(attrs, risk_obs)

        events1 = contract.get_events()
        events2 = contract.get_events(force_regenerate=True)

        # Should be different objects (regenerated)
        assert events1 is not events2
        # But should have same content
        assert len(events1.events) == len(events2.events)

    def test_get_events_in_range(self):
        """Test filtering events by time range."""
        attrs = ContractAttributes(
            contract_id="TEST",
            contract_type=ContractType.PAM,
            contract_role=ContractRole.RPA,
            status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
        )
        risk_obs = ConstantRiskFactorObserver(1.0)
        contract = MockContract(attrs, risk_obs, num_events=5)

        # Get events in first 2 months
        start = ActusDateTime(2024, 1, 1, 0, 0, 0)
        end = ActusDateTime(2024, 2, 28, 23, 59, 59)
        filtered = contract.get_events_in_range(start, end)

        assert len(filtered) <= 5
        # All filtered events should be in range
        for event in filtered:
            assert event.event_time >= start
            assert event.event_time <= end


class TestContractSimulation:
    """Test contract simulation."""

    def test_simulate_generates_history(self):
        """Test that simulate generates simulation history."""
        attrs = ContractAttributes(
            contract_id="TEST",
            contract_type=ContractType.PAM,
            contract_role=ContractRole.RPA,
            status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
            notional_principal=100000.0,
            nominal_interest_rate=0.05,
        )
        risk_obs = ConstantRiskFactorObserver(1.0)
        contract = MockContract(attrs, risk_obs, num_events=3)

        history = contract.simulate()

        assert isinstance(history, SimulationHistory)
        assert len(history.events) == 3
        assert len(history.states) == 3
        assert isinstance(history.initial_state, ContractState)
        assert isinstance(history.final_state, ContractState)

    def test_simulate_applies_stf_and_pof(self):
        """Test that simulate applies STF and POF."""
        attrs = ContractAttributes(
            contract_id="TEST",
            contract_type=ContractType.PAM,
            contract_role=ContractRole.RPA,
            status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
            notional_principal=100000.0,
        )
        risk_obs = ConstantRiskFactorObserver(1.0)
        contract = MockContract(attrs, risk_obs, num_events=3)

        history = contract.simulate()

        # Check that payoffs were calculated
        for event in history.events:
            assert event.payoff is not None
            assert isinstance(event.payoff, jnp.ndarray)

        # Check that states were updated
        for event in history.events:
            assert event.state_pre is not None
            assert event.state_post is not None

    def test_simulate_updates_status_dates(self):
        """Test that simulation updates status dates."""
        attrs = ContractAttributes(
            contract_id="TEST",
            contract_type=ContractType.PAM,
            contract_role=ContractRole.RPA,
            status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
        )
        risk_obs = ConstantRiskFactorObserver(1.0)
        contract = MockContract(attrs, risk_obs, num_events=3)

        history = contract.simulate()

        # Status date should advance with each event
        for i, event in enumerate(history.events):
            if i > 0:
                prev_sd = history.events[i - 1].state_post.sd
                curr_sd = event.state_post.sd
                assert curr_sd >= prev_sd


class TestCashflowExtraction:
    """Test cashflow extraction methods."""

    def test_get_cashflows_returns_timeline(self):
        """Test that get_cashflows returns cashflow timeline."""
        attrs = ContractAttributes(
            contract_id="TEST",
            contract_type=ContractType.PAM,
            contract_role=ContractRole.RPA,
            status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
        )
        risk_obs = ConstantRiskFactorObserver(1.0)
        contract = MockContract(attrs, risk_obs, num_events=3)

        cashflows = contract.get_cashflows()

        assert len(cashflows) == 3
        for time, payoff, currency in cashflows:
            assert isinstance(time, ActusDateTime)
            assert isinstance(payoff, jnp.ndarray)
            assert isinstance(currency, str)

    def test_simulation_history_get_cashflows(self):
        """Test SimulationHistory.get_cashflows method."""
        attrs = ContractAttributes(
            contract_id="TEST",
            contract_type=ContractType.PAM,
            contract_role=ContractRole.RPA,
            status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
        )
        risk_obs = ConstantRiskFactorObserver(1.0)
        contract = MockContract(attrs, risk_obs, num_events=3)

        history = contract.simulate()
        cashflows = history.get_cashflows()

        assert len(cashflows) == 3
        assert cashflows == [(e.event_time, e.payoff, e.currency) for e in history.events]

    def test_simulation_history_filter_events(self):
        """Test SimulationHistory.filter_events method."""
        attrs = ContractAttributes(
            contract_id="TEST",
            contract_type=ContractType.PAM,
            contract_role=ContractRole.RPA,
            status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
        )
        risk_obs = ConstantRiskFactorObserver(1.0)
        contract = MockContract(attrs, risk_obs, num_events=5)

        history = contract.simulate()

        # Filter to first 2 months
        start = ActusDateTime(2024, 1, 1, 0, 0, 0)
        end = ActusDateTime(2024, 2, 28, 23, 59, 59)
        filtered = history.filter_events(start, end)

        assert len(filtered) <= 5
        for event in filtered:
            assert event.event_time >= start
            assert event.event_time <= end


class TestValidation:
    """Test contract validation."""

    def test_validate_success(self):
        """Test validation of valid contract."""
        attrs = ContractAttributes(
            contract_id="TEST",
            contract_type=ContractType.PAM,
            contract_role=ContractRole.RPA,
            status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
            notional_principal=100000.0,
            nominal_interest_rate=0.05,
        )
        risk_obs = ConstantRiskFactorObserver(1.0)
        contract = MockContract(attrs, risk_obs)

        result = contract.validate()

        assert "errors" in result
        assert "warnings" in result
        assert len(result["errors"]) == 0

    def test_validate_missing_contract_id(self):
        """Test validation catches missing contract_id."""
        attrs = ContractAttributes(
            contract_id="",  # Empty ID
            contract_type=ContractType.PAM,
            contract_role=ContractRole.RPA,
            status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
        )
        risk_obs = ConstantRiskFactorObserver(1.0)
        contract = MockContract(attrs, risk_obs)

        result = contract.validate()

        assert len(result["errors"]) > 0
        assert any("contract_id" in err for err in result["errors"])

    def test_validate_warns_negative_notional(self):
        """Test validation warns about negative notional."""
        attrs = ContractAttributes(
            contract_id="TEST",
            contract_type=ContractType.PAM,
            contract_role=ContractRole.RPA,
            status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
            notional_principal=-100000.0,  # Negative
        )
        risk_obs = ConstantRiskFactorObserver(1.0)
        contract = MockContract(attrs, risk_obs)

        result = contract.validate()

        assert len(result["warnings"]) > 0
        assert any("notional_principal" in warn for warn in result["warnings"])

    def test_validate_warns_high_interest_rate(self):
        """Test validation warns about unusually high interest rate."""
        attrs = ContractAttributes(
            contract_id="TEST",
            contract_type=ContractType.PAM,
            contract_role=ContractRole.RPA,
            status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
            nominal_interest_rate=2.5,  # 250%
        )
        risk_obs = ConstantRiskFactorObserver(1.0)
        contract = MockContract(attrs, risk_obs)

        result = contract.validate()

        assert len(result["warnings"]) > 0
        assert any("interest_rate" in warn for warn in result["warnings"])


class TestHelperFunctions:
    """Test helper functions."""

    def test_sort_events_by_sequence(self):
        """Test sorting events by time and sequence."""
        t1 = ActusDateTime(2024, 1, 1, 0, 0, 0)
        t2 = ActusDateTime(2024, 2, 1, 0, 0, 0)

        event1 = ContractEvent(
            event_type=EventType.IED,
            event_time=t2,
            payoff=jnp.array(0.0),
            currency="USD",
            sequence=0,
        )
        event2 = ContractEvent(
            event_type=EventType.IP,
            event_time=t1,
            payoff=jnp.array(100.0),
            currency="USD",
            sequence=0,
        )
        event3 = ContractEvent(
            event_type=EventType.IP,
            event_time=t1,
            payoff=jnp.array(100.0),
            currency="USD",
            sequence=1,
        )

        sorted_events = sort_events_by_sequence([event1, event2, event3])

        assert sorted_events[0] == event2  # t1, seq 0
        assert sorted_events[1] == event3  # t1, seq 1
        assert sorted_events[2] == event1  # t2, seq 0

    def test_sort_events_is_pure(self):
        """Test that sort_events doesn't modify input."""
        t1 = ActusDateTime(2024, 1, 1, 0, 0, 0)
        event1 = ContractEvent(
            event_type=EventType.IED,
            event_time=t1,
            payoff=jnp.array(0.0),
            currency="USD",
        )
        event2 = ContractEvent(
            event_type=EventType.IP,
            event_time=t1,
            payoff=jnp.array(100.0),
            currency="USD",
        )

        original = [event2, event1]
        original_order = [e.event_type for e in original]

        sort_events_by_sequence(original)

        # Original list should be unchanged
        assert [e.event_type for e in original] == original_order

    def test_merge_scheduled_and_observed_events(self):
        """Test merging scheduled and observed events."""
        t1 = ActusDateTime(2024, 1, 1, 0, 0, 0)
        t2 = ActusDateTime(2024, 2, 1, 0, 0, 0)

        scheduled = [
            ContractEvent(
                event_type=EventType.IED,
                event_time=t1,
                payoff=jnp.array(0.0),
                currency="USD",
            )
        ]
        observed = [
            ContractEvent(
                event_type=EventType.IP,
                event_time=t2,
                payoff=jnp.array(100.0),
                currency="USD",
            )
        ]

        merged = merge_scheduled_and_observed_events(scheduled, observed)

        assert len(merged) == 2
        assert merged[0].event_time == t1
        assert merged[1].event_time == t2

    def test_merge_removes_duplicates(self):
        """Test that merge removes duplicate events."""
        t1 = ActusDateTime(2024, 1, 1, 0, 0, 0)

        event1 = ContractEvent(
            event_type=EventType.IP,
            event_time=t1,
            payoff=jnp.array(100.0),
            currency="USD",
        )
        event2 = ContractEvent(
            event_type=EventType.IP,
            event_time=t1,
            payoff=jnp.array(200.0),  # Different payoff
            currency="USD",
        )

        merged = merge_scheduled_and_observed_events([event1], [event2])

        # Should keep only first occurrence
        assert len(merged) == 1
        assert merged[0].payoff == jnp.array(100.0)
