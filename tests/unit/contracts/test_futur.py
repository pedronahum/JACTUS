"""Unit tests for FUTUR (Future) contract."""

import pytest

from jactus.contracts import FutureContract, create_contract
from jactus.core import (
    ActusDateTime,
    ContractAttributes,
    ContractRole,
    ContractState,
    ContractType,
    EventType,
)
from jactus.observers import ConstantRiskFactorObserver


class TestFutureInitialization:
    """Test FUTUR contract initialization and validation."""

    def test_future_contract_creation(self):
        """Test successful FUTUR contract creation."""
        attrs = ContractAttributes(
            contract_id="FUTUR001",
            contract_type=ContractType.FUTUR,
            contract_role=ContractRole.RPA,
            status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
            contract_deal_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
            maturity_date=ActusDateTime(2024, 12, 31, 0, 0, 0),
            notional_principal=1.0,
            future_price=100.0,  # PFUT
            contract_structure="AAPL",  # Underlier reference
            purchase_date=ActusDateTime(2024, 1, 15, 0, 0, 0),
            settlement_date=ActusDateTime(2025, 1, 15, 0, 0, 0),
        )

        rf_obs = ConstantRiskFactorObserver(constant_value=105.0)
        contract = FutureContract(attrs, rf_obs)

        assert contract is not None
        assert contract.attributes.contract_type == ContractType.FUTUR
        assert contract.attributes.future_price == 100.0

    def test_future_requires_future_price(self):
        """Test that FUTUR contract requires future_price (PFUT)."""
        attrs = ContractAttributes(
            contract_id="FUTUR001",
            contract_type=ContractType.FUTUR,
            contract_role=ContractRole.RPA,
            status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
            contract_deal_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
            maturity_date=ActusDateTime(2024, 12, 31, 0, 0, 0),
            notional_principal=1.0,
            # Missing future_price
            contract_structure="AAPL",
        )

        rf_obs = ConstantRiskFactorObserver(constant_value=105.0)

        with pytest.raises(ValueError, match="future_price.*required"):
            FutureContract(attrs, rf_obs)

    def test_future_requires_contract_structure(self):
        """Test that FUTUR contract requires contract_structure (CTST)."""
        attrs = ContractAttributes(
            contract_id="FUTUR001",
            contract_type=ContractType.FUTUR,
            contract_role=ContractRole.RPA,
            status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
            contract_deal_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
            maturity_date=ActusDateTime(2024, 12, 31, 0, 0, 0),
            notional_principal=1.0,
            future_price=100.0,
            # Missing contract_structure
        )

        rf_obs = ConstantRiskFactorObserver(constant_value=105.0)

        with pytest.raises(ValueError, match="contract_structure.*required"):
            FutureContract(attrs, rf_obs)


class TestFutureEventSchedule:
    """Test FUTUR event schedule generation."""

    def test_future_event_schedule(self):
        """Test that FUTUR generates correct event schedule."""
        attrs = ContractAttributes(
            contract_id="FUTUR001",
            contract_type=ContractType.FUTUR,
            contract_role=ContractRole.RPA,
            status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
            contract_deal_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
            maturity_date=ActusDateTime(2024, 12, 31, 0, 0, 0),
            notional_principal=1.0,
            future_price=100.0,
            contract_structure="AAPL",
            purchase_date=ActusDateTime(2024, 1, 15, 0, 0, 0),
            settlement_date=ActusDateTime(2025, 1, 15, 0, 0, 0),
        )

        rf_obs = ConstantRiskFactorObserver(constant_value=105.0)
        contract = FutureContract(attrs, rf_obs)

        schedule = contract.generate_event_schedule()

        # Extract event types
        event_types = [event.event_type for event in schedule.events]

        # Should have: PRD, MD, STD (no AD for FUTUR)
        assert EventType.PRD in event_types
        assert EventType.MD in event_types
        assert EventType.STD in event_types

    def test_future_event_schedule_no_purchase(self):
        """Test FUTUR event schedule without purchase date."""
        attrs = ContractAttributes(
            contract_id="FUTUR001",
            contract_type=ContractType.FUTUR,
            contract_role=ContractRole.RPA,
            status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
            contract_deal_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
            maturity_date=ActusDateTime(2024, 12, 31, 0, 0, 0),
            notional_principal=1.0,
            future_price=100.0,
            contract_structure="AAPL",
            settlement_date=ActusDateTime(2025, 1, 15, 0, 0, 0),
            # No purchase_date
        )

        rf_obs = ConstantRiskFactorObserver(constant_value=105.0)
        contract = FutureContract(attrs, rf_obs)

        schedule = contract.generate_event_schedule()
        event_types = [event.event_type for event in schedule.events]

        # Should not have PRD
        assert EventType.PRD not in event_types


class TestFutureStateInitialization:
    """Test FUTUR state initialization."""

    def test_initialize_state(self):
        """Test FUTUR state initialization."""
        attrs = ContractAttributes(
            contract_id="FUTUR001",
            contract_type=ContractType.FUTUR,
            contract_role=ContractRole.RPA,
            status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
            contract_deal_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
            maturity_date=ActusDateTime(2024, 12, 31, 0, 0, 0),
            notional_principal=1.0,
            future_price=100.0,
            contract_structure="AAPL",
        )

        rf_obs = ConstantRiskFactorObserver(constant_value=105.0)
        contract = FutureContract(attrs, rf_obs)

        state = contract.initialize_state()

        assert isinstance(state, ContractState)


class TestFuturePayoffs:
    """Test FUTUR payoff calculations."""

    def test_future_no_premium_at_purchase(self):
        """Test that FUTUR has no premium payment at purchase (PRD)."""
        attrs = ContractAttributes(
            contract_id="FUTUR001",
            contract_type=ContractType.FUTUR,
            contract_role=ContractRole.RPA,
            status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
            contract_deal_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
            maturity_date=ActusDateTime(2024, 12, 31, 0, 0, 0),
            notional_principal=10.0,
            future_price=100.0,
            contract_structure="AAPL",
            purchase_date=ActusDateTime(2024, 1, 15, 0, 0, 0),
        )

        rf_obs = ConstantRiskFactorObserver(constant_value=105.0)
        contract = FutureContract(attrs, rf_obs)

        state = contract.initialize_state()
        pof = contract.get_payoff_function(EventType.PRD)

        purchase_time = attrs.purchase_date
        payoff = pof.calculate_payoff(EventType.PRD, state, attrs, purchase_time, rf_obs)

        # No premium payment for futures
        assert float(payoff) == pytest.approx(0.0, abs=0.01)

    def test_future_positive_settlement(self):
        """Test FUTUR settlement when spot > futures price (long profit)."""
        attrs = ContractAttributes(
            contract_id="FUTUR001",
            contract_type=ContractType.FUTUR,
            contract_role=ContractRole.RPA,
            status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
            contract_deal_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
            maturity_date=ActusDateTime(2024, 12, 31, 0, 0, 0),
            notional_principal=10.0,
            future_price=100.0,  # Agreed futures price
            contract_structure="AAPL",
            settlement_date=ActusDateTime(2025, 1, 15, 0, 0, 0),
        )

        # Spot price at maturity = 110 (above futures price)
        rf_obs = ConstantRiskFactorObserver(constant_value=110.0)
        contract = FutureContract(attrs, rf_obs)

        # Simulate to maturity
        state = contract.initialize_state()

        # Apply MD state transition
        stf = contract.get_state_transition_function(EventType.MD)
        maturity_time = attrs.maturity_date
        state_md = stf.transition_state(EventType.MD, state, attrs, maturity_time, rf_obs)

        # Calculate STD payoff
        pof = contract.get_payoff_function(EventType.STD)
        settlement_time = attrs.settlement_date
        payoff = pof.calculate_payoff(EventType.STD, state_md, attrs, settlement_time, rf_obs)

        # Payoff = (Spot - Futures) * Notional = (110 - 100) * 10 = 100
        assert float(payoff) == pytest.approx(100.0, abs=0.01)

    def test_future_negative_settlement(self):
        """Test FUTUR settlement when spot < futures price (long loss)."""
        attrs = ContractAttributes(
            contract_id="FUTUR001",
            contract_type=ContractType.FUTUR,
            contract_role=ContractRole.RPA,
            status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
            contract_deal_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
            maturity_date=ActusDateTime(2024, 12, 31, 0, 0, 0),
            notional_principal=10.0,
            future_price=100.0,  # Agreed futures price
            contract_structure="AAPL",
            settlement_date=ActusDateTime(2025, 1, 15, 0, 0, 0),
        )

        # Spot price at maturity = 90 (below futures price)
        rf_obs = ConstantRiskFactorObserver(constant_value=90.0)
        contract = FutureContract(attrs, rf_obs)

        # Simulate to maturity
        state = contract.initialize_state()

        # Apply MD state transition
        stf = contract.get_state_transition_function(EventType.MD)
        maturity_time = attrs.maturity_date
        state_md = stf.transition_state(EventType.MD, state, attrs, maturity_time, rf_obs)

        # Calculate STD payoff
        pof = contract.get_payoff_function(EventType.STD)
        settlement_time = attrs.settlement_date
        payoff = pof.calculate_payoff(EventType.STD, state_md, attrs, settlement_time, rf_obs)

        # Payoff = (Spot - Futures) * Notional = (90 - 100) * 10 = -100 (loss)
        assert float(payoff) == pytest.approx(-100.0, abs=0.01)

    def test_future_zero_settlement(self):
        """Test FUTUR settlement when spot = futures price (no profit/loss)."""
        attrs = ContractAttributes(
            contract_id="FUTUR001",
            contract_type=ContractType.FUTUR,
            contract_role=ContractRole.RPA,
            status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
            contract_deal_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
            maturity_date=ActusDateTime(2024, 12, 31, 0, 0, 0),
            notional_principal=10.0,
            future_price=100.0,  # Agreed futures price
            contract_structure="AAPL",
            settlement_date=ActusDateTime(2025, 1, 15, 0, 0, 0),
        )

        # Spot price at maturity = 100 (equals futures price)
        rf_obs = ConstantRiskFactorObserver(constant_value=100.0)
        contract = FutureContract(attrs, rf_obs)

        # Simulate to maturity
        state = contract.initialize_state()

        # Apply MD state transition
        stf = contract.get_state_transition_function(EventType.MD)
        maturity_time = attrs.maturity_date
        state_md = stf.transition_state(EventType.MD, state, attrs, maturity_time, rf_obs)

        # Calculate STD payoff
        pof = contract.get_payoff_function(EventType.STD)
        settlement_time = attrs.settlement_date
        payoff = pof.calculate_payoff(EventType.STD, state_md, attrs, settlement_time, rf_obs)

        # Payoff = (Spot - Futures) * Notional = (100 - 100) * 10 = 0
        assert float(payoff) == pytest.approx(0.0, abs=0.01)


class TestFutureStateTransitions:
    """Test FUTUR state transition functions."""

    def test_future_maturity_state_transition(self):
        """Test FUTUR maturity state transition calculates settlement amount."""
        attrs = ContractAttributes(
            contract_id="FUTUR001",
            contract_type=ContractType.FUTUR,
            contract_role=ContractRole.RPA,
            status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
            contract_deal_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
            maturity_date=ActusDateTime(2024, 12, 31, 0, 0, 0),
            notional_principal=1.0,
            future_price=100.0,
            contract_structure="AAPL",
        )

        # Spot at maturity = 110
        rf_obs = ConstantRiskFactorObserver(constant_value=110.0)
        contract = FutureContract(attrs, rf_obs)

        state = contract.initialize_state()

        # Apply MD state transition
        stf = contract.get_state_transition_function(EventType.MD)
        maturity_time = attrs.maturity_date
        state_md = stf.transition_state(EventType.MD, state, attrs, maturity_time, rf_obs)

        # Settlement amount should be Spot - Futures = 110 - 100 = 10
        assert float(state_md.xa) == pytest.approx(10.0, abs=0.01)

    def test_future_maturity_negative_settlement(self):
        """Test FUTUR maturity with negative settlement amount."""
        attrs = ContractAttributes(
            contract_id="FUTUR001",
            contract_type=ContractType.FUTUR,
            contract_role=ContractRole.RPA,
            status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
            contract_deal_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
            maturity_date=ActusDateTime(2024, 12, 31, 0, 0, 0),
            notional_principal=1.0,
            future_price=100.0,
            contract_structure="AAPL",
        )

        # Spot at maturity = 85
        rf_obs = ConstantRiskFactorObserver(constant_value=85.0)
        contract = FutureContract(attrs, rf_obs)

        state = contract.initialize_state()

        # Apply MD state transition
        stf = contract.get_state_transition_function(EventType.MD)
        maturity_time = attrs.maturity_date
        state_md = stf.transition_state(EventType.MD, state, attrs, maturity_time, rf_obs)

        # Settlement amount should be Spot - Futures = 85 - 100 = -15
        assert float(state_md.xa) == pytest.approx(-15.0, abs=0.01)


class TestFutureSimulation:
    """Test complete FUTUR simulation."""

    def test_simulate_commodity_future(self):
        """Test complete simulation of a commodity future (gold)."""
        attrs = ContractAttributes(
            contract_id="FUTUR_GC001",
            contract_type=ContractType.FUTUR,
            contract_role=ContractRole.RPA,
            status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
            contract_deal_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
            maturity_date=ActusDateTime(2024, 6, 30, 0, 0, 0),
            notional_principal=100.0,  # 100 oz of gold
            future_price=1800.0,  # Locked-in price per oz
            contract_structure="GC",  # Gold commodity code
            purchase_date=ActusDateTime(2024, 1, 15, 0, 0, 0),
            settlement_date=ActusDateTime(2024, 7, 15, 0, 0, 0),
        )

        # Gold spot at maturity = $1850/oz
        rf_obs = ConstantRiskFactorObserver(constant_value=1850.0)
        contract = FutureContract(attrs, rf_obs)

        # Run simulation
        cashflows = contract.simulate(rf_obs)

        # Should have cashflows for: AD, PRD (0), MD, STD
        assert len(cashflows.events) >= 2

        # Find STD cashflow (final settlement)
        std_cf = next((cf for cf in cashflows.events if cf.event_type == EventType.STD), None)
        assert std_cf is not None

        # Settlement = (1850 - 1800) * 100 = 5000 profit
        assert float(std_cf.payoff) == pytest.approx(5000.0, abs=0.01)

    def test_simulate_index_future(self):
        """Test complete simulation of an index future (S&P 500)."""
        attrs = ContractAttributes(
            contract_id="FUTUR_SPX001",
            contract_type=ContractType.FUTUR,
            contract_role=ContractRole.RPA,
            status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
            contract_deal_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
            maturity_date=ActusDateTime(2024, 3, 31, 0, 0, 0),
            notional_principal=50.0,  # $50 multiplier
            future_price=4500.0,  # Index level
            contract_structure="SPX",  # S&P 500 index
            settlement_date=ActusDateTime(2024, 4, 15, 0, 0, 0),
        )

        # Index at maturity = 4400 (down 100 points)
        rf_obs = ConstantRiskFactorObserver(constant_value=4400.0)
        contract = FutureContract(attrs, rf_obs)

        # Run simulation
        cashflows = contract.simulate(rf_obs)

        # Find STD cashflow
        std_cf = next((cf for cf in cashflows.events if cf.event_type == EventType.STD), None)
        assert std_cf is not None

        # Settlement = (4400 - 4500) * 50 = -5000 (loss)
        assert float(std_cf.payoff) == pytest.approx(-5000.0, abs=0.01)

    def test_simulate_stock_future(self):
        """Test simulation of a single-stock future."""
        attrs = ContractAttributes(
            contract_id="FUTUR_AAPL001",
            contract_type=ContractType.FUTUR,
            contract_role=ContractRole.RPA,
            status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
            contract_deal_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
            maturity_date=ActusDateTime(2024, 12, 31, 0, 0, 0),
            notional_principal=100.0,  # 100 shares
            future_price=150.0,  # Locked-in price per share
            contract_structure="AAPL",
            settlement_date=ActusDateTime(2025, 1, 15, 0, 0, 0),
        )

        # Stock at maturity = $155
        rf_obs = ConstantRiskFactorObserver(constant_value=155.0)
        contract = FutureContract(attrs, rf_obs)

        # Run simulation
        cashflows = contract.simulate(rf_obs)

        # Find STD cashflow
        std_cf = next((cf for cf in cashflows.events if cf.event_type == EventType.STD), None)
        assert std_cf is not None

        # Settlement = (155 - 150) * 100 = 500 profit
        assert float(std_cf.payoff) == pytest.approx(500.0, abs=0.01)


class TestFutureFactory:
    """Test FUTUR creation via factory."""

    def test_create_future_via_factory(self):
        """Test creating FUTUR contract using create_contract factory."""
        attrs = ContractAttributes(
            contract_id="FUTUR001",
            contract_type=ContractType.FUTUR,
            contract_role=ContractRole.RPA,
            status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
            contract_deal_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
            maturity_date=ActusDateTime(2024, 12, 31, 0, 0, 0),
            notional_principal=1.0,
            future_price=100.0,
            contract_structure="AAPL",
        )

        rf_obs = ConstantRiskFactorObserver(constant_value=105.0)

        # Create via factory
        contract = create_contract(attrs, rf_obs)

        # Should be a FutureContract instance
        assert isinstance(contract, FutureContract)
        assert contract.attributes.contract_type == ContractType.FUTUR

        # Should be able to simulate
        cashflows = contract.simulate(rf_obs)
        assert len(cashflows.events) >= 2


class TestFutureEdgeCases:
    """Test edge cases and boundary conditions for FUTUR."""

    def test_future_large_price_movement(self):
        """Test FUTUR with very large price movement."""
        attrs = ContractAttributes(
            contract_id="FUTUR001",
            contract_type=ContractType.FUTUR,
            contract_role=ContractRole.RPA,
            status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
            contract_deal_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
            maturity_date=ActusDateTime(2024, 12, 31, 0, 0, 0),
            notional_principal=1.0,
            future_price=100.0,
            contract_structure="CRYPTO",
            settlement_date=ActusDateTime(2025, 1, 15, 0, 0, 0),
        )

        # Extreme price movement (3x increase)
        rf_obs = ConstantRiskFactorObserver(constant_value=300.0)
        contract = FutureContract(attrs, rf_obs)

        state = contract.initialize_state()
        stf = contract.get_state_transition_function(EventType.MD)
        state_md = stf.transition_state(EventType.MD, state, attrs, attrs.maturity_date, rf_obs)

        # Settlement should be 300 - 100 = 200
        assert float(state_md.xa) == pytest.approx(200.0, abs=0.01)

    def test_future_same_purchase_and_maturity(self):
        """Test FUTUR where purchase and maturity are same day."""
        attrs = ContractAttributes(
            contract_id="FUTUR001",
            contract_type=ContractType.FUTUR,
            contract_role=ContractRole.RPA,
            status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
            contract_deal_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
            maturity_date=ActusDateTime(2024, 1, 15, 0, 0, 0),
            notional_principal=1.0,
            future_price=100.0,
            contract_structure="AAPL",
            purchase_date=ActusDateTime(2024, 1, 15, 0, 0, 0),
            settlement_date=ActusDateTime(2024, 1, 16, 0, 0, 0),
        )

        rf_obs = ConstantRiskFactorObserver(constant_value=105.0)
        contract = FutureContract(attrs, rf_obs)

        # Should create contract successfully
        assert contract is not None

        # Should have both PRD and MD events
        schedule = contract.generate_event_schedule()
        event_types = [event.event_type for event in schedule.events]
        assert EventType.PRD in event_types
        assert EventType.MD in event_types


class TestFutureEventPayoffs:
    """Test all FUTUR event type payoffs for coverage."""

    def test_ad_event_payoff(self):
        """Test AD (Analysis Date) event has zero payoff."""
        attrs = ContractAttributes(
            contract_id="FUTUR001",
            contract_type=ContractType.FUTUR,
            contract_role=ContractRole.RPA,
            status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
            contract_deal_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
            maturity_date=ActusDateTime(2024, 12, 31, 0, 0, 0),
            notional_principal=1.0,
            future_price=100.0,
            contract_structure="AAPL",
        )

        rf_obs = ConstantRiskFactorObserver(constant_value=105.0)
        contract = FutureContract(attrs, rf_obs)

        state = contract.initialize_state()
        pof = contract.get_payoff_function(EventType.AD)

        payoff = pof.calculate_payoff(EventType.AD, state, attrs, attrs.status_date, rf_obs)

        # AD event has zero payoff
        assert float(payoff) == pytest.approx(0.0, abs=0.01)

    def test_ied_event_payoff(self):
        """Test IED (Initial Exchange Date) event has zero payoff."""
        attrs = ContractAttributes(
            contract_id="FUTUR001",
            contract_type=ContractType.FUTUR,
            contract_role=ContractRole.RPA,
            status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
            contract_deal_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
            initial_exchange_date=ActusDateTime(2024, 1, 2, 0, 0, 0),
            maturity_date=ActusDateTime(2024, 12, 31, 0, 0, 0),
            notional_principal=1.0,
            future_price=100.0,
            contract_structure="AAPL",
        )

        rf_obs = ConstantRiskFactorObserver(constant_value=105.0)
        contract = FutureContract(attrs, rf_obs)

        state = contract.initialize_state()
        pof = contract.get_payoff_function(EventType.IED)

        payoff = pof.calculate_payoff(
            EventType.IED, state, attrs, attrs.initial_exchange_date, rf_obs
        )

        # IED not used for FUTUR, zero payoff
        assert float(payoff) == pytest.approx(0.0, abs=0.01)

    def test_md_event_payoff(self):
        """Test MD (Maturity Date) event has zero payoff."""
        attrs = ContractAttributes(
            contract_id="FUTUR001",
            contract_type=ContractType.FUTUR,
            contract_role=ContractRole.RPA,
            status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
            contract_deal_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
            maturity_date=ActusDateTime(2024, 12, 31, 0, 0, 0),
            notional_principal=1.0,
            future_price=100.0,
            contract_structure="AAPL",
        )

        rf_obs = ConstantRiskFactorObserver(constant_value=105.0)
        contract = FutureContract(attrs, rf_obs)

        state = contract.initialize_state()
        pof = contract.get_payoff_function(EventType.MD)

        payoff = pof.calculate_payoff(EventType.MD, state, attrs, attrs.maturity_date, rf_obs)

        # MD has zero payoff (settlement happens at STD)
        assert float(payoff) == pytest.approx(0.0, abs=0.01)

    def test_ce_event_payoff(self):
        """Test CE (Credit Event) event has zero payoff."""
        attrs = ContractAttributes(
            contract_id="FUTUR001",
            contract_type=ContractType.FUTUR,
            contract_role=ContractRole.RPA,
            status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
            contract_deal_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
            maturity_date=ActusDateTime(2024, 12, 31, 0, 0, 0),
            notional_principal=1.0,
            future_price=100.0,
            contract_structure="AAPL",
        )

        rf_obs = ConstantRiskFactorObserver(constant_value=105.0)
        contract = FutureContract(attrs, rf_obs)

        state = contract.initialize_state()
        pof = contract.get_payoff_function(EventType.CE)

        ce_time = ActusDateTime(2024, 6, 1, 0, 0, 0)
        payoff = pof.calculate_payoff(EventType.CE, state, attrs, ce_time, rf_obs)

        # CE has zero payoff
        assert float(payoff) == pytest.approx(0.0, abs=0.01)


class TestFutureStateTransitionCoverage:
    """Test FUTUR state transitions for coverage."""

    def test_ad_state_transition(self):
        """Test AD state transition."""
        attrs = ContractAttributes(
            contract_id="FUTUR001",
            contract_type=ContractType.FUTUR,
            contract_role=ContractRole.RPA,
            status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
            contract_deal_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
            maturity_date=ActusDateTime(2024, 12, 31, 0, 0, 0),
            notional_principal=1.0,
            future_price=100.0,
            contract_structure="AAPL",
        )

        rf_obs = ConstantRiskFactorObserver(constant_value=105.0)
        contract = FutureContract(attrs, rf_obs)

        state = contract.initialize_state()
        stf = contract.get_state_transition_function(EventType.AD)

        state_ad = stf.transition_state(EventType.AD, state, attrs, attrs.status_date, rf_obs)

        # AD doesn't change state
        assert isinstance(state_ad, ContractState)

    def test_ied_state_transition(self):
        """Test IED state transition."""
        attrs = ContractAttributes(
            contract_id="FUTUR001",
            contract_type=ContractType.FUTUR,
            contract_role=ContractRole.RPA,
            status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
            contract_deal_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
            initial_exchange_date=ActusDateTime(2024, 1, 2, 0, 0, 0),
            maturity_date=ActusDateTime(2024, 12, 31, 0, 0, 0),
            notional_principal=1.0,
            future_price=100.0,
            contract_structure="AAPL",
        )

        rf_obs = ConstantRiskFactorObserver(constant_value=105.0)
        contract = FutureContract(attrs, rf_obs)

        state = contract.initialize_state()
        stf = contract.get_state_transition_function(EventType.IED)

        state_ied = stf.transition_state(
            EventType.IED, state, attrs, attrs.initial_exchange_date, rf_obs
        )

        # IED doesn't change state for FUTUR
        assert isinstance(state_ied, ContractState)

    def test_prd_state_transition(self):
        """Test PRD state transition."""
        attrs = ContractAttributes(
            contract_id="FUTUR001",
            contract_type=ContractType.FUTUR,
            contract_role=ContractRole.RPA,
            status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
            contract_deal_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
            maturity_date=ActusDateTime(2024, 12, 31, 0, 0, 0),
            notional_principal=1.0,
            future_price=100.0,
            contract_structure="AAPL",
            purchase_date=ActusDateTime(2024, 1, 15, 0, 0, 0),
        )

        rf_obs = ConstantRiskFactorObserver(constant_value=105.0)
        contract = FutureContract(attrs, rf_obs)

        state = contract.initialize_state()
        stf = contract.get_state_transition_function(EventType.PRD)

        state_prd = stf.transition_state(EventType.PRD, state, attrs, attrs.purchase_date, rf_obs)

        # PRD doesn't change state
        assert isinstance(state_prd, ContractState)

    def test_std_state_transition(self):
        """Test STD state transition."""
        attrs = ContractAttributes(
            contract_id="FUTUR001",
            contract_type=ContractType.FUTUR,
            contract_role=ContractRole.RPA,
            status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
            contract_deal_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
            maturity_date=ActusDateTime(2024, 12, 31, 0, 0, 0),
            notional_principal=1.0,
            future_price=100.0,
            contract_structure="AAPL",
            settlement_date=ActusDateTime(2025, 1, 15, 0, 0, 0),
        )

        rf_obs = ConstantRiskFactorObserver(constant_value=105.0)
        contract = FutureContract(attrs, rf_obs)

        state = contract.initialize_state()
        stf = contract.get_state_transition_function(EventType.STD)

        state_std = stf.transition_state(EventType.STD, state, attrs, attrs.settlement_date, rf_obs)

        # STD just returns state (settlement calculated at MD)
        assert isinstance(state_std, ContractState)

    def test_ce_state_transition(self):
        """Test CE state transition."""
        attrs = ContractAttributes(
            contract_id="FUTUR001",
            contract_type=ContractType.FUTUR,
            contract_role=ContractRole.RPA,
            status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
            contract_deal_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
            maturity_date=ActusDateTime(2024, 12, 31, 0, 0, 0),
            notional_principal=1.0,
            future_price=100.0,
            contract_structure="AAPL",
        )

        rf_obs = ConstantRiskFactorObserver(constant_value=105.0)
        contract = FutureContract(attrs, rf_obs)

        state = contract.initialize_state()
        stf = contract.get_state_transition_function(EventType.CE)

        ce_time = ActusDateTime(2024, 6, 1, 0, 0, 0)
        state_ce = stf.transition_state(EventType.CE, state, attrs, ce_time, rf_obs)

        # CE doesn't change state
        assert isinstance(state_ce, ContractState)


class TestFutureUnknownEvent:
    """Test FUTUR with unknown event type."""

    def test_unknown_event_type_payoff(self):
        """Test that unknown event types return zero payoff."""
        attrs = ContractAttributes(
            contract_id="FUTUR001",
            contract_type=ContractType.FUTUR,
            contract_role=ContractRole.RPA,
            status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
            contract_deal_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
            maturity_date=ActusDateTime(2024, 12, 31, 0, 0, 0),
            notional_principal=1.0,
            future_price=100.0,
            contract_structure="AAPL",
        )

        rf_obs = ConstantRiskFactorObserver(constant_value=105.0)
        contract = FutureContract(attrs, rf_obs)

        state = contract.initialize_state()
        pof = contract.get_payoff_function(EventType.IP)  # IP not supported

        # Use calculate_payoff with unknown event type
        payoff = pof.calculate_payoff(EventType.IP, state, attrs, attrs.status_date, rf_obs)

        # Unknown event should return zero
        assert float(payoff) == pytest.approx(0.0, abs=0.01)


class TestFutureValidation:
    """Test FUTUR validation edge cases."""

    def test_wrong_contract_type_raises(self):
        """Test that wrong contract type raises ValueError."""
        attrs = ContractAttributes(
            contract_id="FUTUR001",
            contract_type=ContractType.PAM,  # Wrong type!
            contract_role=ContractRole.RPA,
            status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
            contract_deal_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
            maturity_date=ActusDateTime(2024, 12, 31, 0, 0, 0),
            notional_principal=1.0,
            future_price=100.0,
            contract_structure="AAPL",
        )

        rf_obs = ConstantRiskFactorObserver(constant_value=105.0)

        with pytest.raises(ValueError, match="Expected contract_type=FUTUR"):
            FutureContract(attrs, rf_obs)

    def test_missing_maturity_date_raises(self):
        """Test that missing maturity_date raises ValueError."""
        attrs = ContractAttributes(
            contract_id="FUTUR001",
            contract_type=ContractType.FUTUR,
            contract_role=ContractRole.RPA,
            status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
            contract_deal_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
            # Missing maturity_date
            notional_principal=1.0,
            future_price=100.0,
            contract_structure="AAPL",
        )

        rf_obs = ConstantRiskFactorObserver(constant_value=105.0)

        with pytest.raises(ValueError, match="maturity_date is required"):
            FutureContract(attrs, rf_obs)


class TestFutureAnalysisDates:
    """Test FUTUR with analysis dates."""

    def test_future_with_analysis_dates(self):
        """Test FUTUR event schedule includes analysis dates."""
        attrs = ContractAttributes(
            contract_id="FUTUR001",
            contract_type=ContractType.FUTUR,
            contract_role=ContractRole.RPA,
            status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
            contract_deal_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
            maturity_date=ActusDateTime(2024, 12, 31, 0, 0, 0),
            notional_principal=1.0,
            future_price=100.0,
            contract_structure="AAPL",
            analysis_dates=[
                ActusDateTime(2024, 3, 31, 0, 0, 0),
                ActusDateTime(2024, 6, 30, 0, 0, 0),
                ActusDateTime(2024, 9, 30, 0, 0, 0),
            ],
        )

        rf_obs = ConstantRiskFactorObserver(constant_value=105.0)
        contract = FutureContract(attrs, rf_obs)

        schedule = contract.generate_event_schedule()
        event_types = [event.event_type for event in schedule.events]

        # Should have AD events
        ad_events = [e for e in schedule.events if e.event_type == EventType.AD]
        assert len(ad_events) == 3

    def test_future_default_performance_status(self):
        """Test FUTUR uses default performance status when not specified."""
        attrs = ContractAttributes(
            contract_id="FUTUR001",
            contract_type=ContractType.FUTUR,
            contract_role=ContractRole.RPA,
            status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
            contract_deal_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
            maturity_date=ActusDateTime(2024, 12, 31, 0, 0, 0),
            notional_principal=1.0,
            future_price=100.0,
            contract_structure="AAPL",
            # Not setting contract_performance - should default to PF
        )

        rf_obs = ConstantRiskFactorObserver(constant_value=105.0)
        contract = FutureContract(attrs, rf_obs)

        # Initialize state should use default PF
        state = contract.initialize_state()
        assert state.prf == "PF"

        # Event schedule should also use default PF
        schedule = contract.generate_event_schedule()
        for event in schedule.events:
            if event.state_post:
                assert event.state_post.prf == "PF"


class TestFutureTermination:
    """Test FUTUR with termination date."""

    def test_future_with_termination(self):
        """Test FUTUR with early termination."""
        attrs = ContractAttributes(
            contract_id="FUTUR001",
            contract_type=ContractType.FUTUR,
            contract_role=ContractRole.RPA,
            status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
            contract_deal_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
            maturity_date=ActusDateTime(2024, 12, 31, 0, 0, 0),
            notional_principal=1.0,
            future_price=100.0,
            contract_structure="AAPL",
            termination_date=ActusDateTime(2024, 6, 30, 0, 0, 0),
            price_at_termination_date=95.0,
        )

        rf_obs = ConstantRiskFactorObserver(constant_value=105.0)
        contract = FutureContract(attrs, rf_obs)

        state = contract.initialize_state()

        # Test TD payoff
        pof = contract.get_payoff_function(EventType.TD)
        td_payoff = pof.calculate_payoff(EventType.TD, state, attrs, attrs.termination_date, rf_obs)

        # TD payoff = price_at_termination_date * notional = 95 * 1 = 95 (positive)
        assert float(td_payoff) == pytest.approx(95.0, abs=0.01)

        # Test TD state transition
        stf = contract.get_state_transition_function(EventType.TD)
        state_td = stf.transition_state(EventType.TD, state, attrs, attrs.termination_date, rf_obs)

        # TD sets contract to terminated
        assert isinstance(state_td, ContractState)
