"""Tests for CLM (Call Money) contract implementation."""

import pytest

from jactus.contracts import create_contract
from jactus.contracts.clm import CallMoneyContract, CLMPayoffFunction, CLMStateTransitionFunction
from jactus.core import (
    ActusDateTime,
    ContractAttributes,
    ContractRole,
    ContractState,
    ContractType,
    DayCountConvention,
    EventType,
)
from jactus.observers import ConstantRiskFactorObserver


class TestCLMInitialization:
    """Test CLM contract initialization."""

    def test_clm_basic_initialization(self):
        """Test basic CLM initialization."""
        attrs = ContractAttributes(
            contract_id="CLM-TEST-001",
            contract_type=ContractType.CLM,
            contract_role=ContractRole.RPA,
            status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
            initial_exchange_date=ActusDateTime(2024, 1, 15, 0, 0, 0),
            # No maturity_date - will be determined dynamically
            currency="USD",
            notional_principal=50000.0,
            nominal_interest_rate=0.08,
            day_count_convention=DayCountConvention.A360,
        )

        rf_obs = ConstantRiskFactorObserver(constant_value=0.08)
        contract = create_contract(attrs, rf_obs)

        assert isinstance(contract, CallMoneyContract)
        assert contract.attributes.contract_type == ContractType.CLM

    def test_clm_initialization_with_maturity(self):
        """Test CLM initialization with optional maturity date."""
        attrs = ContractAttributes(
            contract_id="CLM-TEST-002",
            contract_type=ContractType.CLM,
            contract_role=ContractRole.RPA,
            status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
            initial_exchange_date=ActusDateTime(2024, 1, 15, 0, 0, 0),
            maturity_date=ActusDateTime(2025, 1, 15, 0, 0, 0),  # Optional
            currency="USD",
            notional_principal=50000.0,
            nominal_interest_rate=0.08,
            day_count_convention=DayCountConvention.A360,
        )

        rf_obs = ConstantRiskFactorObserver(constant_value=0.08)
        contract = create_contract(attrs, rf_obs)

        assert contract is not None
        assert contract.attributes.maturity_date is not None


class TestCLMEventSchedule:
    """Test CLM event schedule generation."""

    def test_clm_schedule_without_maturity(self):
        """Test event schedule when maturity is not set."""
        attrs = ContractAttributes(
            contract_id="CLM-TEST-003",
            contract_type=ContractType.CLM,
            contract_role=ContractRole.RPA,
            status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
            initial_exchange_date=ActusDateTime(2024, 1, 15, 0, 0, 0),
            # No maturity_date
            currency="USD",
            notional_principal=50000.0,
            nominal_interest_rate=0.08,
            day_count_convention=DayCountConvention.A360,
        )

        rf_obs = ConstantRiskFactorObserver(constant_value=0.08)
        contract = create_contract(attrs, rf_obs)
        schedule = contract.generate_event_schedule()

        # Should have AD and IED but no MD/IP
        event_types = [e.event_type for e in schedule.events]
        assert EventType.AD in event_types
        assert EventType.IED in event_types
        assert EventType.MD not in event_types  # No fixed maturity
        assert EventType.IP not in event_types  # No IP without maturity

    def test_clm_schedule_with_maturity(self):
        """Test event schedule when maturity is set."""
        attrs = ContractAttributes(
            contract_id="CLM-TEST-004",
            contract_type=ContractType.CLM,
            contract_role=ContractRole.RPA,
            status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
            initial_exchange_date=ActusDateTime(2024, 1, 15, 0, 0, 0),
            maturity_date=ActusDateTime(2025, 1, 15, 0, 0, 0),
            currency="USD",
            notional_principal=50000.0,
            nominal_interest_rate=0.08,
            day_count_convention=DayCountConvention.A360,
        )

        rf_obs = ConstantRiskFactorObserver(constant_value=0.08)
        contract = create_contract(attrs, rf_obs)
        schedule = contract.generate_event_schedule()

        # Should have AD, IED, IP (at maturity), and MD
        event_types = [e.event_type for e in schedule.events]
        assert EventType.AD in event_types
        assert EventType.IED in event_types
        assert EventType.IP in event_types  # Single IP at maturity
        assert EventType.MD in event_types

        # IP should be at same time as MD
        ip_events = [e for e in schedule.events if e.event_type == EventType.IP]
        md_events = [e for e in schedule.events if e.event_type == EventType.MD]
        assert len(ip_events) == 1
        assert len(md_events) == 1
        assert ip_events[0].event_time == md_events[0].event_time

    def test_clm_ipci_schedule(self):
        """Test IPCI (interest capitalization) schedule."""
        attrs = ContractAttributes(
            contract_id="CLM-TEST-005",
            contract_type=ContractType.CLM,
            contract_role=ContractRole.RPA,
            status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
            initial_exchange_date=ActusDateTime(2024, 1, 15, 0, 0, 0),
            maturity_date=ActusDateTime(2025, 1, 15, 0, 0, 0),
            currency="USD",
            notional_principal=50000.0,
            nominal_interest_rate=0.08,
            day_count_convention=DayCountConvention.A360,
            interest_payment_cycle="3M",  # Quarterly capitalization
            interest_capitalization_end_date=ActusDateTime(2025, 1, 15, 0, 0, 0),
        )

        rf_obs = ConstantRiskFactorObserver(constant_value=0.08)
        contract = create_contract(attrs, rf_obs)
        schedule = contract.generate_event_schedule()

        # Should have IPCI events
        ipci_events = [e for e in schedule.events if e.event_type == EventType.IPCI]
        assert len(ipci_events) > 0  # Should have quarterly IPCI events

    def test_clm_rate_reset_schedule(self):
        """Test rate reset schedule."""
        attrs = ContractAttributes(
            contract_id="CLM-TEST-006",
            contract_type=ContractType.CLM,
            contract_role=ContractRole.RPA,
            status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
            initial_exchange_date=ActusDateTime(2024, 1, 15, 0, 0, 0),
            maturity_date=ActusDateTime(2025, 1, 15, 0, 0, 0),
            currency="USD",
            notional_principal=50000.0,
            nominal_interest_rate=0.08,
            day_count_convention=DayCountConvention.A360,
            rate_reset_cycle="6M",  # Semi-annual rate resets
        )

        rf_obs = ConstantRiskFactorObserver(constant_value=0.08)
        contract = create_contract(attrs, rf_obs)
        schedule = contract.generate_event_schedule()

        # Should have RR events
        rr_events = [e for e in schedule.events if e.event_type == EventType.RR]
        assert len(rr_events) > 0  # Should have semi-annual RR events


class TestCLMStateTransitions:
    """Test CLM state transition functions."""

    def test_ied_initializes_state(self):
        """Test that IED event initializes contract state."""
        attrs = ContractAttributes(
            contract_id="CLM-TEST-007",
            contract_type=ContractType.CLM,
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

        stf = CLMStateTransitionFunction()
        rf_obs = ConstantRiskFactorObserver(constant_value=0.08)

        new_state = stf.transition_state(
            EventType.IED, state, attrs, ActusDateTime(2024, 1, 15, 0, 0, 0), rf_obs
        )

        # State should be initialized
        assert float(new_state.nt) == pytest.approx(50000.0, abs=1.0)
        assert float(new_state.ipnr) == pytest.approx(0.08, abs=0.001)
        assert float(new_state.ipac) == 0.0

    def test_ipci_capitalizes_interest(self):
        """Test that IPCI event capitalizes interest to notional."""
        attrs = ContractAttributes(
            contract_id="CLM-TEST-008",
            contract_type=ContractType.CLM,
            contract_role=ContractRole.RPA,
            status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
            initial_exchange_date=ActusDateTime(2024, 1, 15, 0, 0, 0),
            currency="USD",
            notional_principal=50000.0,
            nominal_interest_rate=0.08,
            day_count_convention=DayCountConvention.A360,
        )

        # State after some time with accrued interest
        state = ContractState(
            sd=ActusDateTime(2024, 1, 15, 0, 0, 0),
            tmd=None,
            nt=50000.0,
            ipnr=0.08,
            ipac=500.0,  # Some accrued interest
            feac=0.0,
            nsc=1.0,
            isc=1.0,
        )

        stf = CLMStateTransitionFunction()
        rf_obs = ConstantRiskFactorObserver(constant_value=0.08)

        # IPCI after 3 months
        new_state = stf.transition_state(
            EventType.IPCI,
            state,
            attrs,
            ActusDateTime(2024, 4, 15, 0, 0, 0),
            rf_obs,
        )

        # Notional should increase (interest capitalized)
        assert float(new_state.nt) > 50000.0
        # Accrued interest should be reset
        assert float(new_state.ipac) == 0.0

    def test_pr_reduces_notional(self):
        """Test that PR event reduces notional."""
        attrs = ContractAttributes(
            contract_id="CLM-TEST-009",
            contract_type=ContractType.CLM,
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

        stf = CLMStateTransitionFunction()
        rf_obs = ConstantRiskFactorObserver(constant_value=0.08)

        # PR event (principal repayment)
        new_state = stf.transition_state(
            EventType.PR, state, attrs, ActusDateTime(2024, 6, 15, 0, 0, 0), rf_obs
        )

        # Notional should be reduced (to zero for full repayment)
        assert float(new_state.nt) == pytest.approx(0.0, abs=1.0)

    def test_rr_updates_rate(self):
        """Test that RR event updates interest rate."""
        attrs = ContractAttributes(
            contract_id="CLM-TEST-010",
            contract_type=ContractType.CLM,
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

        stf = CLMStateTransitionFunction()
        # Observer with new rate
        rf_obs = ConstantRiskFactorObserver(constant_value=0.10)

        # RR event (rate reset)
        new_state = stf.transition_state(
            EventType.RR, state, attrs, ActusDateTime(2024, 6, 15, 0, 0, 0), rf_obs
        )

        # Rate should be updated
        assert float(new_state.ipnr) == pytest.approx(0.10, abs=0.001)


class TestCLMPayoffs:
    """Test CLM payoff functions."""

    def test_ied_payoff(self):
        """Test IED payoff (disburse principal)."""
        attrs = ContractAttributes(
            contract_id="CLM-TEST-011",
            contract_type=ContractType.CLM,
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

        pof = CLMPayoffFunction(
            contract_role=attrs.contract_role,
            currency=attrs.currency,
        )
        rf_obs = ConstantRiskFactorObserver(constant_value=0.08)

        payoff = pof.calculate_payoff(
            EventType.IED, state, attrs, ActusDateTime(2024, 1, 15, 0, 0, 0), rf_obs
        )

        # IED formula: R(CNTRL) × (-1) × (NT + PDIED) = 1 × (-1) × 50000 = -50000
        assert float(payoff) == pytest.approx(-50000.0, abs=1.0)

    def test_md_payoff_with_interest(self):
        """Test MD payoff (return principal + interest)."""
        attrs = ContractAttributes(
            contract_id="CLM-TEST-012",
            contract_type=ContractType.CLM,
            contract_role=ContractRole.RPA,
            status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
            initial_exchange_date=ActusDateTime(2024, 1, 15, 0, 0, 0),
            currency="USD",
            notional_principal=50000.0,
            nominal_interest_rate=0.08,
            day_count_convention=DayCountConvention.A360,
        )

        # State after 1 year with some accrued interest
        state = ContractState(
            sd=ActusDateTime(2024, 1, 15, 0, 0, 0),
            tmd=ActusDateTime(2025, 1, 15, 0, 0, 0),
            nt=50000.0,
            ipnr=0.08,
            ipac=1000.0,  # Some previously accrued
            feac=0.0,
            nsc=1.0,
            isc=1.0,
        )

        pof = CLMPayoffFunction(
            contract_role=attrs.contract_role,
            currency=attrs.currency,
        )
        rf_obs = ConstantRiskFactorObserver(constant_value=0.08)

        payoff = pof.calculate_payoff(
            EventType.MD, state, attrs, ActusDateTime(2025, 1, 15, 0, 0, 0), rf_obs
        )

        # MD payoff returns principal only: R(CNTRL) × Nsc × Nt = 1 × 1 × 50000
        # Interest is paid separately by the IP event at maturity
        assert float(payoff) == pytest.approx(50000.0, abs=1.0)

    def test_ipci_has_no_payoff(self):
        """Test that IPCI event has no payoff."""
        attrs = ContractAttributes(
            contract_id="CLM-TEST-013",
            contract_type=ContractType.CLM,
            contract_role=ContractRole.RPA,
            status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
            initial_exchange_date=ActusDateTime(2024, 1, 15, 0, 0, 0),
            currency="USD",
            notional_principal=50000.0,
            nominal_interest_rate=0.08,
        )

        state = ContractState(
            sd=ActusDateTime(2024, 1, 15, 0, 0, 0),
            tmd=None,
            nt=50000.0,
            ipnr=0.08,
            ipac=500.0,
            feac=0.0,
            nsc=1.0,
            isc=1.0,
        )

        pof = CLMPayoffFunction(
            contract_role=attrs.contract_role,
            currency=attrs.currency,
        )
        rf_obs = ConstantRiskFactorObserver(constant_value=0.08)

        payoff = pof.calculate_payoff(
            EventType.IPCI, state, attrs, ActusDateTime(2024, 4, 15, 0, 0, 0), rf_obs
        )

        # IPCI has no payoff (interest is capitalized, not paid)
        assert float(payoff) == 0.0


class TestCLMSimulation:
    """Test complete CLM simulations."""

    def test_simple_clm_simulation_with_maturity(self):
        """Test complete simulation with fixed maturity."""
        attrs = ContractAttributes(
            contract_id="CLM-SIM-001",
            contract_type=ContractType.CLM,
            contract_role=ContractRole.RPA,
            status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
            initial_exchange_date=ActusDateTime(2024, 1, 15, 0, 0, 0),
            maturity_date=ActusDateTime(2025, 1, 15, 0, 0, 0),
            currency="USD",
            notional_principal=50000.0,
            nominal_interest_rate=0.08,
            day_count_convention=DayCountConvention.A360,
        )

        rf_obs = ConstantRiskFactorObserver(constant_value=0.08)
        contract = create_contract(attrs, rf_obs)

        result = contract.simulate()

        # Simulation should complete
        assert len(result.events) > 0

        # Should have IED, IP, and MD events
        event_types = [e.event_type for e in result.events]
        assert EventType.IED in event_types
        assert EventType.IP in event_types
        assert EventType.MD in event_types

    def test_clm_with_ipci_simulation(self):
        """Test simulation with interest capitalization."""
        attrs = ContractAttributes(
            contract_id="CLM-SIM-002",
            contract_type=ContractType.CLM,
            contract_role=ContractRole.RPA,
            status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
            initial_exchange_date=ActusDateTime(2024, 1, 15, 0, 0, 0),
            maturity_date=ActusDateTime(2025, 1, 15, 0, 0, 0),
            currency="USD",
            notional_principal=50000.0,
            nominal_interest_rate=0.08,
            day_count_convention=DayCountConvention.A360,
            interest_payment_cycle="3M",
            interest_capitalization_end_date=ActusDateTime(2025, 1, 15, 0, 0, 0),
        )

        rf_obs = ConstantRiskFactorObserver(constant_value=0.08)
        contract = create_contract(attrs, rf_obs)

        result = contract.simulate()

        # Should have IPCI events
        ipci_events = [e for e in result.events if e.event_type == EventType.IPCI]
        assert len(ipci_events) > 0

        # Final state should have notional > initial (due to capitalization)
        # (Note: this depends on implementation details of how IPCI affects notional)
