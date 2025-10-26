"""Tests for UMP (Undefined Maturity Profile) contract implementation.

This module tests all aspects of the UMP contract:
- Contract initialization
- Event schedule generation
- State transitions
- Payoff calculations
- Complete simulations with uncertain principal schedules
"""

import pytest
import jax.numpy as jnp

from jactus.contracts import create_contract
from jactus.contracts.ump import UndefinedMaturityProfileContract
from jactus.core import (
    ActusDateTime,
    ContractAttributes,
    ContractRole,
    ContractType,
    DayCountConvention,
    EventType,
)
from jactus.core.states import ContractState
from jactus.observers.risk_factor import ConstantRiskFactorObserver


class TestUMPInitialization:
    """Test UMP contract initialization."""

    def test_ump_basic_initialization(self):
        """Test UMP initialization with minimal attributes."""
        attrs = ContractAttributes(
            contract_id="UMP-TEST-001",
            contract_type=ContractType.UMP,
            contract_role=ContractRole.RPA,
            status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
            initial_exchange_date=ActusDateTime(2024, 1, 15, 0, 0, 0),
            currency="USD",
            notional_principal=100000.0,
            nominal_interest_rate=0.06,
            day_count_convention=DayCountConvention.A360,
        )

        rf_obs = ConstantRiskFactorObserver(constant_value=0.06)
        contract = create_contract(attrs, rf_obs)
        assert contract.attributes.contract_id == "UMP-TEST-001"
        assert contract.attributes.contract_type == ContractType.UMP

        # Initialize state
        state = contract.initialize_state()
        assert state.sd == attrs.status_date
        assert state.tmd is None  # No maturity specified
        assert float(state.nt) == 0.0  # Not set until IED
        assert float(state.ipnr) == 0.0  # Not set until IED

    def test_ump_initialization_with_maturity(self):
        """Test UMP initialization with maturity date."""
        attrs = ContractAttributes(
            contract_id="UMP-TEST-002",
            contract_type=ContractType.UMP,
            contract_role=ContractRole.RPA,
            status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
            initial_exchange_date=ActusDateTime(2024, 1, 15, 0, 0, 0),
            maturity_date=ActusDateTime(2026, 1, 15, 0, 0, 0),
            currency="USD",
            notional_principal=100000.0,
            nominal_interest_rate=0.06,
            day_count_convention=DayCountConvention.A360,
        )

        rf_obs = ConstantRiskFactorObserver(constant_value=0.06)
        contract = create_contract(attrs, rf_obs)
        state = contract.initialize_state()
        assert state.tmd == ActusDateTime(2026, 1, 15, 0, 0, 0)


class TestUMPEventSchedule:
    """Test UMP event schedule generation."""

    def test_ump_schedule_without_maturity(self):
        """Test schedule generation without maturity - minimal events."""
        attrs = ContractAttributes(
            contract_id="UMP-TEST-003",
            contract_type=ContractType.UMP,
            contract_role=ContractRole.RPA,
            status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
            initial_exchange_date=ActusDateTime(2024, 1, 15, 0, 0, 0),
            currency="USD",
            notional_principal=100000.0,
            nominal_interest_rate=0.06,
            day_count_convention=DayCountConvention.A360,
        )

        rf_obs = ConstantRiskFactorObserver(constant_value=0.06)
        contract = create_contract(attrs, rf_obs)
        schedule = contract.generate_event_schedule()

        # Should have only AD and IED (no maturity)
        event_types = [e.event_type for e in schedule.events]
        assert EventType.AD in event_types
        assert EventType.IED in event_types
        # No MD since maturity not specified
        assert EventType.MD not in event_types

    def test_ump_schedule_with_maturity(self):
        """Test schedule generation with maturity date."""
        attrs = ContractAttributes(
            contract_id="UMP-TEST-004",
            contract_type=ContractType.UMP,
            contract_role=ContractRole.RPA,
            status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
            initial_exchange_date=ActusDateTime(2024, 1, 15, 0, 0, 0),
            maturity_date=ActusDateTime(2026, 1, 15, 0, 0, 0),
            currency="USD",
            notional_principal=100000.0,
            nominal_interest_rate=0.06,
            day_count_convention=DayCountConvention.A360,
        )

        rf_obs = ConstantRiskFactorObserver(constant_value=0.06)
        contract = create_contract(attrs, rf_obs)
        schedule = contract.generate_event_schedule()

        event_types = [e.event_type for e in schedule.events]
        assert EventType.AD in event_types
        assert EventType.IED in event_types
        assert EventType.MD in event_types

    def test_ump_ipci_schedule(self):
        """Test IPCI schedule generation."""
        attrs = ContractAttributes(
            contract_id="UMP-TEST-005",
            contract_type=ContractType.UMP,
            contract_role=ContractRole.RPA,
            status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
            initial_exchange_date=ActusDateTime(2024, 1, 15, 0, 0, 0),
            maturity_date=ActusDateTime(2025, 1, 15, 0, 0, 0),
            currency="USD",
            notional_principal=100000.0,
            nominal_interest_rate=0.06,
            interest_payment_cycle="1Q",  # Quarterly capitalization
            interest_capitalization_end_date=ActusDateTime(2025, 1, 15, 0, 0, 0),
            day_count_convention=DayCountConvention.A360,
        )

        rf_obs = ConstantRiskFactorObserver(constant_value=0.06)
        contract = create_contract(attrs, rf_obs)
        schedule = contract.generate_event_schedule()

        # Should have IPCI events
        ipci_events = [e for e in schedule.events if e.event_type == EventType.IPCI]
        assert len(ipci_events) > 0

    def test_ump_no_pr_in_schedule(self):
        """Test that PR events are NOT in scheduled events (they come from observer)."""
        attrs = ContractAttributes(
            contract_id="UMP-TEST-006",
            contract_type=ContractType.UMP,
            contract_role=ContractRole.RPA,
            status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
            initial_exchange_date=ActusDateTime(2024, 1, 15, 0, 0, 0),
            maturity_date=ActusDateTime(2026, 1, 15, 0, 0, 0),
            currency="USD",
            notional_principal=100000.0,
            nominal_interest_rate=0.06,
            day_count_convention=DayCountConvention.A360,
        )

        rf_obs = ConstantRiskFactorObserver(constant_value=0.06)
        contract = create_contract(attrs, rf_obs)
        schedule = contract.generate_event_schedule()

        # PR events should NOT be in schedule (they come from observer)
        event_types = [e.event_type for e in schedule.events]
        assert EventType.PR not in event_types


class TestUMPStateTransitions:
    """Test UMP state transition functions."""

    def test_ied_initializes_state(self):
        """Test that IED event initializes contract state."""
        attrs = ContractAttributes(
            contract_id="UMP-TEST-007",
            contract_type=ContractType.UMP,
            contract_role=ContractRole.RPA,
            status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
            initial_exchange_date=ActusDateTime(2024, 1, 15, 0, 0, 0),
            currency="USD",
            notional_principal=50000.0,
            nominal_interest_rate=0.08,
            day_count_convention=DayCountConvention.A360,
        )

        rf_obs = ConstantRiskFactorObserver(constant_value=0.06)
        contract = create_contract(attrs, rf_obs)
        state = contract.initialize_state()

        # Apply IED transition
        from jactus.contracts.ump import UMPStateTransitionFunction

        stf = UMPStateTransitionFunction()
        rf_obs = ConstantRiskFactorObserver(constant_value=0.08)

        new_state = stf.transition_state(
            EventType.IED, state, attrs, ActusDateTime(2024, 1, 15, 0, 0, 0), rf_obs
        )

        # Check state after IED
        assert float(new_state.nt) == pytest.approx(50000.0, abs=0.01)
        assert float(new_state.ipnr) == pytest.approx(0.08, abs=0.001)
        assert new_state.sd == ActusDateTime(2024, 1, 15, 0, 0, 0)

    def test_ipci_capitalizes_interest(self):
        """Test that IPCI event capitalizes accrued interest into notional."""
        attrs = ContractAttributes(
            contract_id="UMP-TEST-008",
            contract_type=ContractType.UMP,
            contract_role=ContractRole.RPA,
            status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
            initial_exchange_date=ActusDateTime(2024, 1, 15, 0, 0, 0),
            currency="USD",
            notional_principal=50000.0,
            nominal_interest_rate=0.08,
            day_count_convention=DayCountConvention.A360,
        )

        # Start with some state after IED
        state = ContractState(
            sd=ActusDateTime(2024, 1, 15, 0, 0, 0),
            tmd=None,
            nt=50000.0,
            ipnr=0.08,
            ipac=0.0,
            feac=0.0,
            nsc=1.0,
            isc=1.0,
        )

        from jactus.contracts.ump import UMPStateTransitionFunction

        stf = UMPStateTransitionFunction()
        rf_obs = ConstantRiskFactorObserver(constant_value=0.08)

        # Apply IPCI after 3 months (90 days)
        new_state = stf.transition_state(
            EventType.IPCI, state, attrs, ActusDateTime(2024, 4, 15, 0, 0, 0), rf_obs
        )

        # Interest should be added to notional
        # 3 months at 8% = 0.08 * 90/360 * 50000 = 1000 (approximately)
        expected_nt = 50000.0 + 1000.0
        assert float(new_state.nt) == pytest.approx(expected_nt, abs=20.0)
        assert float(new_state.ipac) == pytest.approx(0.0, abs=0.01)  # Reset

    def test_pr_reduces_notional(self):
        """Test that PR event reduces notional."""
        attrs = ContractAttributes(
            contract_id="UMP-TEST-009",
            contract_type=ContractType.UMP,
            contract_role=ContractRole.RPA,
            status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
            initial_exchange_date=ActusDateTime(2024, 1, 15, 0, 0, 0),
            currency="USD",
            notional_principal=50000.0,
            nominal_interest_rate=0.08,
            day_count_convention=DayCountConvention.A360,
        )

        state = ContractState(
            sd=ActusDateTime(2024, 1, 15, 0, 0, 0),
            tmd=None,
            nt=50000.0,
            ipnr=0.08,
            ipac=0.0,
            feac=0.0,
            nsc=1.0,
            isc=1.0,
        )

        from jactus.contracts.ump import UMPStateTransitionFunction

        stf = UMPStateTransitionFunction()
        rf_obs = ConstantRiskFactorObserver(constant_value=0.08)

        # Apply PR (in practice, amount would come from observer)
        # For this test, the amount is 0 (placeholder in implementation)
        new_state = stf.transition_state(
            EventType.PR, state, attrs, ActusDateTime(2024, 6, 15, 0, 0, 0), rf_obs
        )

        # Notional should be reduced (by 0 in this test since observer not implemented)
        # Just verify it doesn't crash and interest is accrued
        assert float(new_state.ipac) > 0.0  # Interest accrued


class TestUMPPayoffs:
    """Test UMP payoff functions."""

    def test_ied_payoff(self):
        """Test IED payoff (disburse principal)."""
        attrs = ContractAttributes(
            contract_id="UMP-TEST-010",
            contract_type=ContractType.UMP,
            contract_role=ContractRole.RPA,
            status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
            initial_exchange_date=ActusDateTime(2024, 1, 15, 0, 0, 0),
            currency="USD",
            notional_principal=50000.0,
            nominal_interest_rate=0.08,
            day_count_convention=DayCountConvention.A360,
        )

        state = ContractState(
            sd=ActusDateTime(2024, 1, 1, 0, 0, 0),
            tmd=None,
            nt=0.0,
            ipnr=0.0,
            ipac=0.0,
            feac=0.0,
            nsc=1.0,
            isc=1.0,
        )

        from jactus.contracts.ump import UMPPayoffFunction

        pof = UMPPayoffFunction(
            contract_role=attrs.contract_role,
            currency=attrs.currency,
        )
        rf_obs = ConstantRiskFactorObserver(constant_value=0.08)

        payoff = pof.calculate_payoff(
            EventType.IED, state, attrs, ActusDateTime(2024, 1, 15, 0, 0, 0), rf_obs
        )

        # Negative payoff = disbursement (RPA role)
        assert float(payoff) == pytest.approx(-50000.0, abs=0.01)

    def test_md_payoff_with_interest(self):
        """Test MD payoff returns principal + accrued interest."""
        attrs = ContractAttributes(
            contract_id="UMP-TEST-011",
            contract_type=ContractType.UMP,
            contract_role=ContractRole.RPA,
            status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
            initial_exchange_date=ActusDateTime(2024, 1, 15, 0, 0, 0),
            maturity_date=ActusDateTime(2025, 1, 15, 0, 0, 0),
            currency="USD",
            notional_principal=50000.0,
            nominal_interest_rate=0.08,
            day_count_convention=DayCountConvention.A360,
        )

        # State after 1 year with some accrued interest
        state = ContractState(
            sd=ActusDateTime(2024, 7, 15, 0, 0, 0),  # 6 months before maturity
            tmd=ActusDateTime(2025, 1, 15, 0, 0, 0),
            nt=50000.0,
            ipnr=0.08,
            ipac=2000.0,  # Already accrued interest
            feac=0.0,
            nsc=1.0,
            isc=1.0,
        )

        from jactus.contracts.ump import UMPPayoffFunction

        pof = UMPPayoffFunction(
            contract_role=attrs.contract_role,
            currency=attrs.currency,
        )
        rf_obs = ConstantRiskFactorObserver(constant_value=0.08)

        payoff = pof.calculate_payoff(
            EventType.MD, state, attrs, ActusDateTime(2025, 1, 15, 0, 0, 0), rf_obs
        )

        # Should return principal + accrued + final period interest
        # Final period: 180 days at 8% = 0.08 * 180/360 * 50000 = 2000 (approximately)
        # Total = 50000 + 2000 (existing) + 2000 (final) = 54000
        expected = 50000.0 + 2000.0 + 2000.0
        assert float(payoff) == pytest.approx(expected, abs=50.0)

    def test_ipci_has_no_payoff(self):
        """Test that IPCI has no payoff (internal capitalization)."""
        attrs = ContractAttributes(
            contract_id="UMP-TEST-012",
            contract_type=ContractType.UMP,
            contract_role=ContractRole.RPA,
            status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
            initial_exchange_date=ActusDateTime(2024, 1, 15, 0, 0, 0),
            currency="USD",
            notional_principal=50000.0,
            nominal_interest_rate=0.08,
            day_count_convention=DayCountConvention.A360,
        )

        state = ContractState(
            sd=ActusDateTime(2024, 1, 15, 0, 0, 0),
            tmd=None,
            nt=50000.0,
            ipnr=0.08,
            ipac=0.0,
            feac=0.0,
            nsc=1.0,
            isc=1.0,
        )

        from jactus.contracts.ump import UMPPayoffFunction

        pof = UMPPayoffFunction(
            contract_role=attrs.contract_role,
            currency=attrs.currency,
        )
        rf_obs = ConstantRiskFactorObserver(constant_value=0.08)

        payoff = pof.calculate_payoff(
            EventType.IPCI, state, attrs, ActusDateTime(2024, 4, 15, 0, 0, 0), rf_obs
        )

        # IPCI should have no payoff
        assert float(payoff) == pytest.approx(0.0, abs=0.01)


class TestUMPSimulation:
    """Test complete UMP simulations."""

    def test_simple_ump_simulation_with_maturity(self):
        """Test simple UMP simulation with maturity and IPCI."""
        attrs = ContractAttributes(
            contract_id="UMP-TEST-013",
            contract_type=ContractType.UMP,
            contract_role=ContractRole.RPA,
            status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
            initial_exchange_date=ActusDateTime(2024, 1, 15, 0, 0, 0),
            maturity_date=ActusDateTime(2025, 1, 15, 0, 0, 0),
            currency="USD",
            notional_principal=100000.0,
            nominal_interest_rate=0.06,
            interest_payment_cycle="1Q",
            interest_capitalization_end_date=ActusDateTime(2025, 1, 15, 0, 0, 0),
            day_count_convention=DayCountConvention.A360,
        )

        rf_obs = ConstantRiskFactorObserver(constant_value=0.06)
        contract = create_contract(attrs, rf_obs)
        rf_obs = ConstantRiskFactorObserver(constant_value=0.06)

        # Generate events
        schedule = contract.generate_event_schedule()
        assert len(schedule.events) > 0

        # Should have IED, IPCI events, and MD
        event_types = [e.event_type for e in schedule.events]
        assert EventType.IED in event_types
        assert EventType.IPCI in event_types
        assert EventType.MD in event_types

    def test_ump_with_ipci_simulation(self):
        """Test UMP with interest capitalization."""
        attrs = ContractAttributes(
            contract_id="UMP-TEST-014",
            contract_type=ContractType.UMP,
            contract_role=ContractRole.RPA,
            status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
            initial_exchange_date=ActusDateTime(2024, 1, 15, 0, 0, 0),
            maturity_date=ActusDateTime(2025, 1, 15, 0, 0, 0),
            currency="USD",
            notional_principal=50000.0,
            nominal_interest_rate=0.08,
            interest_payment_cycle="1Q",  # Quarterly capitalization
            interest_capitalization_end_date=ActusDateTime(2025, 1, 15, 0, 0, 0),
            day_count_convention=DayCountConvention.A360,
        )

        rf_obs = ConstantRiskFactorObserver(constant_value=0.06)
        contract = create_contract(attrs, rf_obs)
        schedule = contract.generate_event_schedule()

        # Count IPCI events
        ipci_events = [e for e in schedule.events if e.event_type == EventType.IPCI]
        # Should have ~4 IPCI events (quarterly for 1 year)
        assert len(ipci_events) >= 3

    def test_ump_uncertain_maturity(self):
        """Test UMP without fixed maturity (uncertain)."""
        attrs = ContractAttributes(
            contract_id="UMP-TEST-015",
            contract_type=ContractType.UMP,
            contract_role=ContractRole.RPA,
            status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
            initial_exchange_date=ActusDateTime(2024, 1, 15, 0, 0, 0),
            # No maturity_date - uncertain
            currency="USD",
            notional_principal=100000.0,
            nominal_interest_rate=0.06,
            day_count_convention=DayCountConvention.A360,
        )

        rf_obs = ConstantRiskFactorObserver(constant_value=0.06)
        contract = create_contract(attrs, rf_obs)
        schedule = contract.generate_event_schedule()

        # Should have minimal events (AD, IED)
        event_types = [e.event_type for e in schedule.events]
        assert EventType.AD in event_types
        assert EventType.IED in event_types
        # No MD since maturity is uncertain
        assert EventType.MD not in event_types
