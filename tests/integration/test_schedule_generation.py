"""Integration tests for contract schedule generation (T3.7).

Tests end-to-end schedule generation for all contract types,
verifying that events are properly sequenced and complete.
"""

import pytest

from jactus.contracts import (
    create_contract,
)
from jactus.core import (
    ActusDateTime,
    ContractAttributes,
    ContractRole,
    ContractType,
    DayCountConvention,
    EventType,
)
from jactus.observers import ConstantRiskFactorObserver


class TestCashScheduleGeneration:
    """Test CSH contract schedule generation."""

    @pytest.fixture
    def rf_obs(self):
        """Risk factor observer fixture."""
        return ConstantRiskFactorObserver(constant_value=0.0)

    def test_csh_minimal_schedule(self, rf_obs):
        """Test CSH generates minimal schedule with AD event."""
        attrs = ContractAttributes(
            contract_id="CSH-INT-001",
            contract_type=ContractType.CSH,
            contract_role=ContractRole.RPA,
            status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
            currency="USD",
            notional_principal=100000.0,
        )

        contract = create_contract(attrs, rf_obs)
        schedule = contract.generate_event_schedule()

        # CSH should have at least one AD event
        assert len(schedule.events) >= 1
        assert schedule.events[0].event_type == EventType.AD

    def test_csh_end_to_end_simulation(self, rf_obs):
        """Test CSH complete simulation workflow."""
        attrs = ContractAttributes(
            contract_id="CSH-INT-002",
            contract_type=ContractType.CSH,
            contract_role=ContractRole.RPA,
            status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
            currency="USD",
            notional_principal=50000.0,
        )

        contract = create_contract(attrs, rf_obs)
        result = contract.simulate()

        # Verify simulation completed
        assert result is not None
        assert len(result.events) >= 1

        # Verify all events have required fields
        for event in result.events:
            assert event.event_type is not None
            assert event.event_time is not None
            assert event.payoff is not None
            assert event.currency == "USD"
            assert event.state_pre is not None
            assert event.state_post is not None

        # Get cashflows
        cashflows = result.get_cashflows()
        assert isinstance(cashflows, list)
        assert len(cashflows) == len(result.events)


class TestPAMScheduleGeneration:
    """Test PAM contract schedule generation."""

    @pytest.fixture
    def rf_obs(self):
        """Risk factor observer fixture."""
        return ConstantRiskFactorObserver(constant_value=0.05)

    def test_pam_basic_bullet_loan(self, rf_obs):
        """Test PAM generates complete schedule for bullet loan."""
        attrs = ContractAttributes(
            contract_id="PAM-INT-001",
            contract_type=ContractType.PAM,
            contract_role=ContractRole.RPA,
            status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
            initial_exchange_date=ActusDateTime(2024, 1, 15, 0, 0, 0),
            maturity_date=ActusDateTime(2026, 1, 15, 0, 0, 0),
            currency="USD",
            notional_principal=100000.0,
            nominal_interest_rate=0.05,
            day_count_convention=DayCountConvention.A360,
            interest_payment_cycle="1Y",
        )

        contract = create_contract(attrs, rf_obs)
        schedule = contract.generate_event_schedule()

        # Should have IED, IP events, and MD
        assert len(schedule.events) >= 3

        # Verify event types present
        event_types = {e.event_type for e in schedule.events}
        assert EventType.IED in event_types
        assert EventType.MD in event_types
        assert EventType.IP in event_types

    def test_pam_complete_simulation(self, rf_obs):
        """Test PAM complete simulation workflow."""
        attrs = ContractAttributes(
            contract_id="PAM-INT-002",
            contract_type=ContractType.PAM,
            contract_role=ContractRole.RPA,
            status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
            initial_exchange_date=ActusDateTime(2024, 1, 15, 0, 0, 0),
            maturity_date=ActusDateTime(2025, 1, 15, 0, 0, 0),
            currency="USD",
            notional_principal=100000.0,
            nominal_interest_rate=0.05,
            day_count_convention=DayCountConvention.A360,
            interest_payment_cycle="6M",
        )

        contract = create_contract(attrs, rf_obs)
        result = contract.simulate()

        # Verify simulation completed
        assert result is not None
        assert len(result.events) >= 3  # IED + 2 IP + MD

        # Verify cashflow sequence makes sense
        cashflows = result.get_cashflows()
        times = [t for t, _, _ in cashflows]
        amounts = [amt for _, amt, _ in cashflows]

        # Times should be monotonically increasing
        assert times == sorted(times)

        # IED should be negative (receiving loan)
        assert float(amounts[0]) < 0

        # MD should be positive (paying back)
        assert float(amounts[-1]) > 0

    def test_pam_event_sequencing(self, rf_obs):
        """Test PAM events are properly sequenced."""
        attrs = ContractAttributes(
            contract_id="PAM-INT-003",
            contract_type=ContractType.PAM,
            contract_role=ContractRole.RPA,
            status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
            initial_exchange_date=ActusDateTime(2024, 1, 15, 0, 0, 0),
            maturity_date=ActusDateTime(2025, 1, 15, 0, 0, 0),
            currency="USD",
            notional_principal=100000.0,
            nominal_interest_rate=0.05,
            day_count_convention=DayCountConvention.A360,
            interest_payment_cycle="3M",
        )

        contract = create_contract(attrs, rf_obs)
        result = contract.simulate()

        # Verify sequence numbers are consecutive
        for i, event in enumerate(result.events):
            assert event.sequence == i

        # Verify times are monotonically increasing
        for i in range(len(result.events) - 1):
            curr_time = result.events[i].event_time
            next_time = result.events[i + 1].event_time
            assert curr_time <= next_time


class TestStockScheduleGeneration:
    """Test STK contract schedule generation."""

    @pytest.fixture
    def rf_obs(self):
        """Risk factor observer fixture."""
        return ConstantRiskFactorObserver(constant_value=150.0)

    def test_stk_purchase_and_sale(self, rf_obs):
        """Test STK generates schedule for purchase and sale."""
        attrs = ContractAttributes(
            contract_id="STK-INT-001",
            contract_type=ContractType.STK,
            contract_role=ContractRole.RPA,
            status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
            purchase_date=ActusDateTime(2024, 1, 15, 0, 0, 0),
            termination_date=ActusDateTime(2024, 12, 31, 0, 0, 0),
            currency="USD",
            price_at_purchase_date=150.0,
            price_at_termination_date=175.0,
        )

        contract = create_contract(attrs, rf_obs)
        schedule = contract.generate_event_schedule()

        # Should have PRD and TD
        assert len(schedule.events) >= 2
        assert schedule.events[0].event_type == EventType.PRD
        assert schedule.events[-1].event_type == EventType.TD

    def test_stk_complete_simulation(self, rf_obs):
        """Test STK complete simulation workflow."""
        attrs = ContractAttributes(
            contract_id="STK-INT-002",
            contract_type=ContractType.STK,
            contract_role=ContractRole.RPA,
            status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
            purchase_date=ActusDateTime(2024, 1, 15, 0, 0, 0),
            termination_date=ActusDateTime(2024, 6, 30, 0, 0, 0),
            currency="USD",
            price_at_purchase_date=100.0,
            price_at_termination_date=120.0,
        )

        contract = create_contract(attrs, rf_obs)
        result = contract.simulate()

        # Verify simulation
        assert result is not None
        assert len(result.events) == 2

        # Calculate profit/loss
        cashflows = result.get_cashflows()
        total_cashflow = sum(float(amt) for _, amt, _ in cashflows)

        # Should show profit (120 - 100 = 20)
        assert total_cashflow > 0


class TestCommodityScheduleGeneration:
    """Test COM contract schedule generation."""

    @pytest.fixture
    def rf_obs(self):
        """Risk factor observer fixture."""
        return ConstantRiskFactorObserver(constant_value=80.0)

    def test_com_purchase_and_sale(self, rf_obs):
        """Test COM generates schedule for purchase and sale."""
        attrs = ContractAttributes(
            contract_id="COM-INT-001",
            contract_type=ContractType.COM,
            contract_role=ContractRole.RPA,
            status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
            purchase_date=ActusDateTime(2024, 1, 15, 0, 0, 0),
            termination_date=ActusDateTime(2024, 12, 31, 0, 0, 0),
            currency="USD",
            price_at_purchase_date=7500.0,
            price_at_termination_date=8200.0,
        )

        contract = create_contract(attrs, rf_obs)
        schedule = contract.generate_event_schedule()

        # Should have PRD and TD
        assert len(schedule.events) >= 2
        assert schedule.events[0].event_type == EventType.PRD
        assert schedule.events[-1].event_type == EventType.TD

    def test_com_complete_simulation(self, rf_obs):
        """Test COM complete simulation workflow."""
        attrs = ContractAttributes(
            contract_id="COM-INT-002",
            contract_type=ContractType.COM,
            contract_role=ContractRole.RPA,
            status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
            purchase_date=ActusDateTime(2024, 2, 1, 0, 0, 0),
            termination_date=ActusDateTime(2024, 8, 1, 0, 0, 0),
            currency="USD",
            price_at_purchase_date=5000.0,
            price_at_termination_date=5500.0,
        )

        contract = create_contract(attrs, rf_obs)
        result = contract.simulate()

        # Verify simulation
        assert result is not None
        assert len(result.events) == 2

        # Calculate profit/loss
        cashflows = result.get_cashflows()
        total_cashflow = sum(float(amt) for _, amt, _ in cashflows)

        # Should show profit (5500 - 5000 = 500)
        assert total_cashflow == 500.0


class TestFactoryIntegration:
    """Test factory pattern integration."""

    @pytest.fixture
    def rf_obs(self):
        """Risk factor observer fixture."""
        return ConstantRiskFactorObserver(constant_value=0.05)

    def test_all_contract_types_can_simulate(self, rf_obs):
        """Test all registered contract types can simulate successfully."""
        test_configs = {
            ContractType.CSH: ContractAttributes(
                contract_id="CSH-FACTORY-001",
                contract_type=ContractType.CSH,
                contract_role=ContractRole.RPA,
                status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
                currency="USD",
                notional_principal=100000.0,
            ),
            ContractType.PAM: ContractAttributes(
                contract_id="PAM-FACTORY-001",
                contract_type=ContractType.PAM,
                contract_role=ContractRole.RPA,
                status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
                initial_exchange_date=ActusDateTime(2024, 1, 15, 0, 0, 0),
                maturity_date=ActusDateTime(2025, 1, 15, 0, 0, 0),
                currency="USD",
                notional_principal=100000.0,
                nominal_interest_rate=0.05,
                day_count_convention=DayCountConvention.A360,
            ),
            ContractType.STK: ContractAttributes(
                contract_id="STK-FACTORY-001",
                contract_type=ContractType.STK,
                contract_role=ContractRole.RPA,
                status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
                purchase_date=ActusDateTime(2024, 1, 15, 0, 0, 0),
                termination_date=ActusDateTime(2024, 12, 31, 0, 0, 0),
                currency="USD",
                price_at_purchase_date=150.0,
                price_at_termination_date=175.0,
            ),
            ContractType.COM: ContractAttributes(
                contract_id="COM-FACTORY-001",
                contract_type=ContractType.COM,
                contract_role=ContractRole.RPA,
                status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
                purchase_date=ActusDateTime(2024, 1, 15, 0, 0, 0),
                termination_date=ActusDateTime(2024, 12, 31, 0, 0, 0),
                currency="USD",
                price_at_purchase_date=7500.0,
                price_at_termination_date=8200.0,
            ),
        }

        for _contract_type, attrs in test_configs.items():
            # Create via factory
            contract = create_contract(attrs, rf_obs)

            # Simulate
            result = contract.simulate()

            # Verify
            assert result is not None
            assert len(result.events) >= 1
            assert all(e.currency == "USD" for e in result.events)
