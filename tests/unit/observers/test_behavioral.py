"""Unit tests for behavioral risk factor observers.

Tests for:
- BehaviorRiskFactorObserver Protocol
- BaseBehaviorRiskFactorObserver ABC
- CalloutEvent dataclass
- PrepaymentSurfaceObserver
- DepositTransactionObserver
- Scenario management
"""

import jax.numpy as jnp
import pytest

from jactus.core import ActusDateTime, ContractAttributes, ContractState
from jactus.core.types import ContractPerformance, ContractRole, ContractType
from jactus.observers import (
    BaseBehaviorRiskFactorObserver,
    BehaviorRiskFactorObserver,
    CalloutEvent,
    ConstantRiskFactorObserver,
    DepositTransactionObserver,
    DictRiskFactorObserver,
    PrepaymentSurfaceObserver,
    Scenario,
)
from jactus.utilities.surface import LabeledSurface2D, Surface2D

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def sample_attributes():
    """Create sample contract attributes for testing."""
    return ContractAttributes(
        contract_id="LOAN-001",
        contract_type=ContractType.PAM,
        contract_role=ContractRole.RPA,
        status_date=ActusDateTime(2024, 1, 1),
        initial_exchange_date=ActusDateTime(2024, 1, 15),
        maturity_date=ActusDateTime(2027, 1, 15),
        notional_principal=100_000.0,
        nominal_interest_rate=0.06,
        currency="USD",
    )


@pytest.fixture
def sample_state():
    """Create sample contract state for testing."""
    return ContractState(
        tmd=ActusDateTime(2027, 1, 15),
        sd=ActusDateTime(2025, 1, 15),
        nt=jnp.array(80_000.0),
        ipnr=jnp.array(0.06),
        ipac=jnp.array(200.0),
        feac=jnp.array(0.0),
        nsc=jnp.array(1.0),
        isc=jnp.array(1.0),
        prf=ContractPerformance.PF,
    )


@pytest.fixture
def prepayment_surface():
    """Create a sample prepayment surface."""
    return Surface2D(
        x_margins=jnp.array([-5.0, 0.0, 1.0, 2.0, 3.0]),
        y_margins=jnp.array([0.0, 1.0, 2.0, 3.0, 5.0]),
        values=jnp.array(
            [
                [0.00, 0.00, 0.00, 0.00, 0.00],
                [0.00, 0.00, 0.01, 0.00, 0.00],
                [0.00, 0.01, 0.02, 0.00, 0.00],
                [0.00, 0.02, 0.05, 0.03, 0.005],
                [0.01, 0.05, 0.10, 0.07, 0.02],
            ]
        ),
    )


# ============================================================================
# CalloutEvent tests
# ============================================================================


class TestCalloutEvent:
    """Test CalloutEvent dataclass."""

    def test_create_callout_event(self):
        """Create a callout event with required fields."""
        event = CalloutEvent(
            model_id="ppm01",
            time=ActusDateTime(2025, 6, 1),
            callout_type="MRD",
        )
        assert event.model_id == "ppm01"
        assert event.time == ActusDateTime(2025, 6, 1)
        assert event.callout_type == "MRD"
        assert event.metadata is None

    def test_create_callout_event_with_metadata(self):
        """Create a callout event with metadata."""
        event = CalloutEvent(
            model_id="deposit-trx",
            time=ActusDateTime(2025, 3, 1),
            callout_type="AFD",
            metadata={"contract_id": "DEPOSIT-001"},
        )
        assert event.metadata == {"contract_id": "DEPOSIT-001"}

    def test_callout_event_is_frozen(self):
        """CalloutEvent is immutable."""
        event = CalloutEvent(
            model_id="ppm01",
            time=ActusDateTime(2025, 6, 1),
            callout_type="MRD",
        )
        with pytest.raises(AttributeError):
            event.model_id = "ppm02"  # type: ignore[misc]


# ============================================================================
# BehaviorRiskFactorObserver Protocol tests
# ============================================================================


class TestBehaviorRiskFactorObserverProtocol:
    """Test BehaviorRiskFactorObserver protocol enforcement."""

    def test_protocol_accepts_valid_implementation(self):
        """Valid implementation is recognized as BehaviorRiskFactorObserver."""

        class ValidBehavior:
            def observe_risk_factor(self, identifier, time, state=None, attributes=None):
                return jnp.array(0.0)

            def observe_event(self, identifier, event_type, time, state=None, attributes=None):
                return None

            def contract_start(self, attributes):
                return []

        assert isinstance(ValidBehavior(), BehaviorRiskFactorObserver)

    def test_protocol_rejects_without_contract_start(self):
        """Objects without contract_start are not BehaviorRiskFactorObservers."""

        class MissingContractStart:
            def observe_risk_factor(self, identifier, time, state=None, attributes=None):
                return jnp.array(0.0)

            def observe_event(self, identifier, event_type, time, state=None, attributes=None):
                return None

        assert not isinstance(MissingContractStart(), BehaviorRiskFactorObserver)


# ============================================================================
# BaseBehaviorRiskFactorObserver tests
# ============================================================================


class ConcreteBehavioralObserver(BaseBehaviorRiskFactorObserver):
    """Concrete implementation for testing."""

    def __init__(self, value: float = 0.05, name: str | None = None):
        super().__init__(name)
        self.value = value

    def _get_risk_factor(self, identifier, time, state, attributes):
        if state is not None:
            return jnp.array(float(state.ipnr) * self.value, dtype=jnp.float32)
        return jnp.array(self.value, dtype=jnp.float32)

    def _get_event_data(self, identifier, event_type, time, state, attributes):
        raise KeyError(f"No event data for '{identifier}'")

    def contract_start(self, attributes):
        return [
            CalloutEvent(
                model_id="test-model",
                time=attributes.status_date,
                callout_type="MRD",
            ),
        ]


class TestBaseBehaviorRiskFactorObserver:
    """Test BaseBehaviorRiskFactorObserver ABC."""

    def test_init_default_name(self):
        """Default name is class name."""
        observer = ConcreteBehavioralObserver()
        assert observer.name == "ConcreteBehavioralObserver"

    def test_init_custom_name(self):
        """Custom name overrides default."""
        observer = ConcreteBehavioralObserver(name="MyModel")
        assert observer.name == "MyModel"

    def test_observe_risk_factor_without_state(self):
        """observe_risk_factor works without state."""
        observer = ConcreteBehavioralObserver(value=0.05)
        result = observer.observe_risk_factor("any", ActusDateTime(2025, 1, 1))
        assert float(result) == pytest.approx(0.05)

    def test_observe_risk_factor_with_state(self, sample_state):
        """observe_risk_factor uses state when provided."""
        observer = ConcreteBehavioralObserver(value=0.5)
        result = observer.observe_risk_factor("any", ActusDateTime(2025, 1, 1), state=sample_state)
        # state.ipnr = 0.06, value = 0.5, result = 0.06 * 0.5 = 0.03
        assert float(result) == pytest.approx(0.03)

    def test_contract_start_returns_events(self, sample_attributes):
        """contract_start returns callout events."""
        observer = ConcreteBehavioralObserver()
        events = observer.contract_start(sample_attributes)
        assert len(events) == 1
        assert events[0].callout_type == "MRD"
        assert events[0].time == sample_attributes.status_date

    def test_satisfies_protocol(self):
        """Concrete implementation satisfies BehaviorRiskFactorObserver protocol."""
        observer = ConcreteBehavioralObserver()
        assert isinstance(observer, BehaviorRiskFactorObserver)


# ============================================================================
# PrepaymentSurfaceObserver tests
# ============================================================================


class TestPrepaymentSurfaceObserver:
    """Test PrepaymentSurfaceObserver."""

    def test_init(self, prepayment_surface):
        """Initialize with surface and defaults."""
        observer = PrepaymentSurfaceObserver(surface=prepayment_surface)
        assert observer.model_id == "prepayment-model"
        assert observer.prepayment_cycle == "6M"

    def test_observe_risk_factor_with_state(
        self, prepayment_surface, sample_state, sample_attributes
    ):
        """observe_risk_factor computes prepayment rate from spread and age."""
        observer = PrepaymentSurfaceObserver(
            surface=prepayment_surface,
            fixed_market_rate=0.04,
        )
        # spread = 0.06 (state.ipnr) - 0.04 (market) = 0.02 (i.e., 2.0 on surface x-axis)
        # age = days from 2024-01-15 to 2025-01-15 = ~1.0 year
        time = ActusDateTime(2025, 1, 15)
        result = observer.observe_risk_factor(
            "prepayment-model", time, state=sample_state, attributes=sample_attributes
        )
        assert float(result) >= 0.0
        assert float(result) <= 0.10

    def test_observe_risk_factor_without_state(self, prepayment_surface):
        """Returns 0 when state is not provided."""
        observer = PrepaymentSurfaceObserver(surface=prepayment_surface)
        result = observer.observe_risk_factor("id", ActusDateTime(2025, 1, 1))
        assert float(result) == 0.0

    def test_observe_risk_factor_without_attributes(self, prepayment_surface, sample_state):
        """Returns 0 when attributes are not provided."""
        observer = PrepaymentSurfaceObserver(surface=prepayment_surface)
        result = observer.observe_risk_factor("id", ActusDateTime(2025, 1, 1), state=sample_state)
        assert float(result) == 0.0

    def test_contract_start_generates_events(self, prepayment_surface, sample_attributes):
        """contract_start generates semi-annual prepayment observation events."""
        observer = PrepaymentSurfaceObserver(
            surface=prepayment_surface,
            prepayment_cycle="6M",
        )
        events = observer.contract_start(sample_attributes)
        # 3 years from 2024-01-15 to 2027-01-15, semi-annual = ~5 events
        assert len(events) >= 4
        assert all(e.callout_type == "MRD" for e in events)
        assert all(e.model_id == "prepayment-model" for e in events)

    def test_contract_start_no_ied(self, prepayment_surface):
        """contract_start returns empty list if no IED."""
        attrs = ContractAttributes(
            contract_id="NO-IED",
            contract_type=ContractType.PAM,
            contract_role=ContractRole.RPA,
            status_date=ActusDateTime(2024, 1, 1),
            maturity_date=ActusDateTime(2027, 1, 15),
            notional_principal=100_000.0,
        )
        observer = PrepaymentSurfaceObserver(surface=prepayment_surface)
        events = observer.contract_start(attrs)
        assert events == []

    def test_with_market_observer(self, prepayment_surface, sample_state, sample_attributes):
        """Uses market observer for reference rate when available."""
        market_obs = DictRiskFactorObserver({"UST-5Y": 0.035})
        observer = PrepaymentSurfaceObserver(
            surface=prepayment_surface,
            market_rate_id="UST-5Y",
            market_observer=market_obs,
        )
        # spread = 0.06 - 0.035 = 0.025 → between 2.0 and 3.0 on x-axis
        time = ActusDateTime(2025, 1, 15)
        result = observer.observe_risk_factor(
            "id", time, state=sample_state, attributes=sample_attributes
        )
        assert float(result) >= 0.0

    def test_satisfies_protocol(self, prepayment_surface):
        """Satisfies BehaviorRiskFactorObserver protocol."""
        observer = PrepaymentSurfaceObserver(surface=prepayment_surface)
        assert isinstance(observer, BehaviorRiskFactorObserver)


# ============================================================================
# DepositTransactionObserver tests
# ============================================================================


class TestDepositTransactionObserver:
    """Test DepositTransactionObserver."""

    def test_init(self):
        """Initialize with transaction schedules."""
        observer = DepositTransactionObserver(
            transactions={
                "DEP-001": [
                    (ActusDateTime(2024, 1, 15), 10000.0),
                    (ActusDateTime(2024, 7, 15), -5000.0),
                ],
            },
        )
        assert observer.model_id == "deposit-trx-model"

    def test_observe_exact_time(self):
        """Returns transaction amount at exact time."""
        observer = DepositTransactionObserver(
            transactions={
                "DEP-001": [
                    (ActusDateTime(2024, 1, 15), 10000.0),
                    (ActusDateTime(2024, 7, 15), -5000.0),
                ],
            },
        )
        result = observer.observe_risk_factor("DEP-001", ActusDateTime(2024, 1, 15))
        assert float(result) == pytest.approx(10000.0)

    def test_observe_exact_time_withdrawal(self):
        """Returns negative amount for withdrawals."""
        observer = DepositTransactionObserver(
            transactions={
                "DEP-001": [
                    (ActusDateTime(2024, 1, 15), 10000.0),
                    (ActusDateTime(2024, 7, 15), -5000.0),
                ],
            },
        )
        result = observer.observe_risk_factor("DEP-001", ActusDateTime(2024, 7, 15))
        assert float(result) == pytest.approx(-5000.0)

    def test_observe_no_scheduled_transaction(self):
        """Returns 0 when no transaction at that time."""
        observer = DepositTransactionObserver(
            transactions={
                "DEP-001": [
                    (ActusDateTime(2024, 1, 15), 10000.0),
                ],
            },
        )
        result = observer.observe_risk_factor("DEP-001", ActusDateTime(2024, 3, 1))
        assert float(result) == 0.0

    def test_observe_unknown_contract(self):
        """Raises KeyError for unknown contract ID."""
        observer = DepositTransactionObserver(transactions={"DEP-001": []})
        with pytest.raises(KeyError, match="DEP-999"):
            observer.observe_risk_factor("DEP-999", ActusDateTime(2024, 1, 1))

    def test_contract_start_generates_events(self):
        """contract_start returns callout events for the contract's transactions."""
        observer = DepositTransactionObserver(
            transactions={
                "LOAN-001": [
                    (ActusDateTime(2024, 3, 1), 5000.0),
                    (ActusDateTime(2024, 6, 1), -2000.0),
                ],
            },
        )
        attrs = ContractAttributes(
            contract_id="LOAN-001",
            contract_type=ContractType.UMP,
            contract_role=ContractRole.RPA,
            status_date=ActusDateTime(2024, 1, 1),
            notional_principal=50_000.0,
        )
        events = observer.contract_start(attrs)
        assert len(events) == 2
        assert all(e.callout_type == "AFD" for e in events)

    def test_contract_start_no_matching_contract(self):
        """contract_start returns empty if contract ID not in transactions."""
        observer = DepositTransactionObserver(
            transactions={"OTHER-001": [(ActusDateTime(2024, 1, 1), 1000.0)]},
        )
        attrs = ContractAttributes(
            contract_id="LOAN-001",
            contract_type=ContractType.UMP,
            contract_role=ContractRole.RPA,
            status_date=ActusDateTime(2024, 1, 1),
            notional_principal=50_000.0,
        )
        events = observer.contract_start(attrs)
        assert events == []

    def test_get_transaction_schedule(self):
        """get_transaction_schedule returns full schedule."""
        observer = DepositTransactionObserver(
            transactions={
                "DEP-001": [
                    (ActusDateTime(2024, 7, 15), -5000.0),
                    (ActusDateTime(2024, 1, 15), 10000.0),
                ],
            },
        )
        schedule = observer.get_transaction_schedule("DEP-001")
        assert len(schedule) == 2
        # Should be sorted by time
        assert schedule[0][0] == ActusDateTime(2024, 1, 15)
        assert schedule[1][0] == ActusDateTime(2024, 7, 15)

    def test_from_labeled_surface(self):
        """Create from LabeledSurface2D."""
        surface = LabeledSurface2D(
            x_labels=["DEP-001", "DEP-002"],
            y_labels=["2024-01-15T00:00:00", "2024-07-15T00:00:00"],
            values=jnp.array(
                [
                    [10000.0, -5000.0],
                    [20000.0, 0.0],
                ]
            ),
        )
        observer = DepositTransactionObserver.from_labeled_surface(surface)
        result = observer.observe_risk_factor("DEP-001", ActusDateTime(2024, 1, 15))
        assert float(result) == pytest.approx(10000.0)

    def test_satisfies_protocol(self):
        """Satisfies BehaviorRiskFactorObserver protocol."""
        observer = DepositTransactionObserver(transactions={})
        assert isinstance(observer, BehaviorRiskFactorObserver)


# ============================================================================
# Scenario tests
# ============================================================================


class TestScenario:
    """Test Scenario management class."""

    def test_create_empty_scenario(self):
        """Create scenario with no observers."""
        scenario = Scenario(scenario_id="test")
        assert scenario.scenario_id == "test"
        assert scenario.market_observers == {}
        assert scenario.behavior_observers == {}

    def test_get_observer_single(self):
        """get_observer returns single market observer directly."""
        obs = ConstantRiskFactorObserver(0.05)
        scenario = Scenario(
            scenario_id="base",
            market_observers={"rates": obs},
        )
        result = scenario.get_observer()
        assert result is obs

    def test_get_observer_composite(self):
        """get_observer returns CompositeRiskFactorObserver for multiple observers."""
        scenario = Scenario(
            scenario_id="base",
            market_observers={
                "rates": ConstantRiskFactorObserver(0.05),
                "fx": DictRiskFactorObserver({"USD/EUR": 1.18}),
            },
        )
        result = scenario.get_observer()
        # Should be a composite
        assert hasattr(result, "observe_risk_factor")

    def test_get_observer_no_observers_raises(self):
        """get_observer raises ValueError when no market observers."""
        scenario = Scenario(scenario_id="empty")
        with pytest.raises(ValueError, match="no market observers"):
            scenario.get_observer()

    def test_get_callout_events(self, sample_attributes):
        """get_callout_events collects from all behavioral observers."""
        behavior = ConcreteBehavioralObserver()
        scenario = Scenario(
            scenario_id="test",
            market_observers={"rates": ConstantRiskFactorObserver(0.0)},
            behavior_observers={"model1": behavior},
        )
        events = scenario.get_callout_events(sample_attributes)
        assert len(events) >= 1

    def test_add_market_observer(self):
        """add_market_observer adds/replaces observer."""
        scenario = Scenario(scenario_id="test")
        obs = ConstantRiskFactorObserver(0.05)
        scenario.add_market_observer("rates", obs)
        assert "rates" in scenario.market_observers

    def test_add_behavior_observer(self):
        """add_behavior_observer adds/replaces observer."""
        scenario = Scenario(scenario_id="test")
        obs = ConcreteBehavioralObserver()
        scenario.add_behavior_observer("model1", obs)
        assert "model1" in scenario.behavior_observers

    def test_list_risk_factors(self):
        """list_risk_factors shows all observer types."""
        scenario = Scenario(
            scenario_id="test",
            market_observers={"rates": ConstantRiskFactorObserver(0.05)},
            behavior_observers={"prepay": ConcreteBehavioralObserver()},
        )
        factors = scenario.list_risk_factors()
        assert factors["rates"] == "ConstantRiskFactorObserver"
        assert factors["prepay"] == "ConcreteBehavioralObserver"
