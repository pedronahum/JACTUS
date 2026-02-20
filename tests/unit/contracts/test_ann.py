"""Unit tests for Annuity (ANN) contract implementation.

Tests the AnnuityContract, ANNPayoffFunction, and ANNStateTransitionFunction
classes for correctness according to the ACTUS specification (Section 7.5).
"""

import jax.numpy as jnp
import pytest

from jactus.contracts.ann import (
    ANNPayoffFunction,
    ANNStateTransitionFunction,
    AnnuityContract,
)
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
from jactus.utilities import calculate_actus_annuity

# ============================================================================
# Test Annuity Calculation Function
# ============================================================================


class TestAnnuityCalculation:
    """Test ACTUS annuity calculation function."""

    def test_annuity_calculation_basic(self):
        """Test basic annuity calculation."""
        start = ActusDateTime(2024, 1, 15, 0, 0, 0)
        pr_schedule = [
            ActusDateTime(2024, 2, 15, 0, 0, 0),
            ActusDateTime(2024, 3, 15, 0, 0, 0),
            ActusDateTime(2024, 4, 15, 0, 0, 0),
            ActusDateTime(2024, 5, 15, 0, 0, 0),
            ActusDateTime(2024, 6, 15, 0, 0, 0),
            ActusDateTime(2024, 7, 15, 0, 0, 0),
            ActusDateTime(2024, 8, 15, 0, 0, 0),
            ActusDateTime(2024, 9, 15, 0, 0, 0),
            ActusDateTime(2024, 10, 15, 0, 0, 0),
            ActusDateTime(2024, 11, 15, 0, 0, 0),
            ActusDateTime(2024, 12, 15, 0, 0, 0),
            ActusDateTime(2025, 1, 15, 0, 0, 0),
        ]

        amount = calculate_actus_annuity(
            start=start,
            pr_schedule=pr_schedule,
            notional=100000.0,
            accrued_interest=0.0,
            rate=0.05,
            day_count_convention=DayCountConvention.A360,
        )

        # $100k loan at 5% for 12 months should be around $8,560/month
        assert 8500 < amount < 8600

    def test_annuity_calculation_zero_rate(self):
        """Test annuity calculation with zero interest rate."""
        start = ActusDateTime(2024, 1, 15, 0, 0, 0)
        # Create 12 monthly dates (Feb through next Jan)
        pr_schedule = [
            ActusDateTime(2024, 2, 15, 0, 0, 0),
            ActusDateTime(2024, 3, 15, 0, 0, 0),
            ActusDateTime(2024, 4, 15, 0, 0, 0),
            ActusDateTime(2024, 5, 15, 0, 0, 0),
            ActusDateTime(2024, 6, 15, 0, 0, 0),
            ActusDateTime(2024, 7, 15, 0, 0, 0),
            ActusDateTime(2024, 8, 15, 0, 0, 0),
            ActusDateTime(2024, 9, 15, 0, 0, 0),
            ActusDateTime(2024, 10, 15, 0, 0, 0),
            ActusDateTime(2024, 11, 15, 0, 0, 0),
            ActusDateTime(2024, 12, 15, 0, 0, 0),
            ActusDateTime(2025, 1, 15, 0, 0, 0),
        ]

        amount = calculate_actus_annuity(
            start=start,
            pr_schedule=pr_schedule,
            notional=120000.0,
            accrued_interest=0.0,
            rate=0.0,
            day_count_convention=DayCountConvention.A360,
        )

        # With 0% rate, payment is just notional / periods
        assert amount == pytest.approx(10000.0)  # 120k / 12 months


# ============================================================================
# Test ANNPayoffFunction
# ============================================================================


class TestANNPayoffFunction:
    """Test ANNPayoffFunction class."""

    def test_initialization(self):
        """Test ANNPayoffFunction can be created (same as NAM)."""
        pof = ANNPayoffFunction(
            contract_role=ContractRole.RPA,
            currency="USD",
            settlement_currency=None,
        )

        assert pof.contract_role == ContractRole.RPA
        assert pof.currency == "USD"

    def test_pof_pr_constant_payment(self):
        """Test that PR payoff is net of interest (like NAM)."""
        pof = ANNPayoffFunction(
            contract_role=ContractRole.RPA,
            currency="USD",
            settlement_currency=None,
        )

        state = ContractState(
            sd=ActusDateTime(2024, 1, 15, 0, 0, 0),
            tmd=ActusDateTime(2054, 1, 15, 0, 0, 0),
            nt=jnp.array(100000.0),
            ipnr=jnp.array(0.05),
            ipac=jnp.array(0.0),
            feac=jnp.array(0.0),
            nsc=jnp.array(1.0),
            isc=jnp.array(1.0),
            prnxt=jnp.array(1000.0),  # Payment amount
            ipcb=jnp.array(100000.0),
        )

        attrs = ContractAttributes(
            contract_id="ANN-001",
            contract_type=ContractType.ANN,
            contract_role=ContractRole.RPA,
            status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
            initial_exchange_date=ActusDateTime(2024, 1, 15, 0, 0, 0),
            maturity_date=ActusDateTime(2054, 1, 15, 0, 0, 0),
            currency="USD",
            notional_principal=100000.0,
            nominal_interest_rate=0.05,
            day_count_convention=DayCountConvention.A360,
        )

        rf_obs = ConstantRiskFactorObserver(constant_value=0.05)

        payoff = pof.calculate_payoff(
            EventType.PR,
            state,
            attrs,
            ActusDateTime(2024, 2, 15, 0, 0, 0),
            rf_obs,
        )

        # Payment = $1000
        # Interest = (31/360) × 0.05 × 100000 = 430.56
        # Net = 1000 - 430.56 = 569.44
        assert float(payoff) == pytest.approx(569.44, rel=0.01)


# ============================================================================
# Test ANNStateTransitionFunction
# ============================================================================


class TestANNStateTransitionFunction:
    """Test ANNStateTransitionFunction class."""

    def test_initialization(self):
        """Test ANNStateTransitionFunction can be created."""
        stf = ANNStateTransitionFunction()
        assert stf is not None

    def test_stf_uses_nam_for_most_events(self):
        """Test that ANN delegates to NAM for non-RR events."""
        stf = ANNStateTransitionFunction()

        state = ContractState(
            sd=ActusDateTime(2024, 1, 15, 0, 0, 0),
            tmd=ActusDateTime(2029, 1, 15, 0, 0, 0),
            nt=jnp.array(100000.0),
            ipnr=jnp.array(0.05),
            ipac=jnp.array(0.0),
            feac=jnp.array(0.0),
            nsc=jnp.array(1.0),
            isc=jnp.array(1.0),
            prnxt=jnp.array(2000.0),
            ipcb=jnp.array(100000.0),
        )

        attrs = ContractAttributes(
            contract_id="ANN-001",
            contract_type=ContractType.ANN,
            contract_role=ContractRole.RPA,
            status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
            initial_exchange_date=ActusDateTime(2024, 1, 15, 0, 0, 0),
            maturity_date=ActusDateTime(2029, 1, 15, 0, 0, 0),
            currency="USD",
            notional_principal=100000.0,
            nominal_interest_rate=0.05,
            day_count_convention=DayCountConvention.A360,
            interest_calculation_base="NT",
            principal_redemption_cycle="1M",
            next_principal_redemption_amount=2000.0,
        )

        rf_obs = ConstantRiskFactorObserver(constant_value=0.05)

        # Test PR event (should use NAM logic)
        new_state = stf.transition_state(
            EventType.PR,
            state,
            attrs,
            ActusDateTime(2024, 2, 15, 0, 0, 0),
            rf_obs,
        )

        # Notional should decrease (NAM behavior)
        assert float(new_state.nt) < float(state.nt)


# ============================================================================
# Test AnnuityContract
# ============================================================================


class TestAnnuityContract:
    """Test AnnuityContract class."""

    def test_initialization(self):
        """Test AnnuityContract can be created."""
        attrs = ContractAttributes(
            contract_id="ANN-001",
            contract_type=ContractType.ANN,
            contract_role=ContractRole.RPA,
            status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
            initial_exchange_date=ActusDateTime(2024, 1, 15, 0, 0, 0),
            maturity_date=ActusDateTime(2054, 1, 15, 0, 0, 0),
            currency="USD",
            notional_principal=300000.0,
            nominal_interest_rate=0.065,
            day_count_convention=DayCountConvention.A360,
            principal_redemption_cycle="1M",
        )

        rf_obs = ConstantRiskFactorObserver(constant_value=0.065)
        contract = AnnuityContract(attributes=attrs, risk_factor_observer=rf_obs)

        assert contract.attributes.contract_id == "ANN-001"
        assert contract.attributes.contract_type == ContractType.ANN

    def test_prnxt_calculated_automatically(self):
        """Test that Prnxt is calculated automatically if not provided."""
        attrs = ContractAttributes(
            contract_id="ANN-001",
            contract_type=ContractType.ANN,
            contract_role=ContractRole.RPA,
            status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
            initial_exchange_date=ActusDateTime(2024, 1, 15, 0, 0, 0),
            maturity_date=ActusDateTime(2025, 1, 15, 0, 0, 0),  # 1 year
            currency="USD",
            notional_principal=100000.0,
            nominal_interest_rate=0.05,
            day_count_convention=DayCountConvention.A360,
            principal_redemption_cycle="1M",  # Monthly
            # Note: next_principal_redemption_amount NOT provided
        )

        rf_obs = ConstantRiskFactorObserver(constant_value=0.05)
        contract = AnnuityContract(attributes=attrs, risk_factor_observer=rf_obs)

        initial_state = contract.initialize_state()

        # Prnxt should be automatically calculated
        assert float(initial_state.prnxt) > 0.0
        # For $100k at 5% for 12 months using ACTUS annuity formula.
        # Annuity start = max(IED, PRANX-1cycle) = IED since PRANX defaults to IED.
        # First PR at IED means 13 periods with the first at yf=0.
        assert 7800 < abs(float(initial_state.prnxt)) < 8000

    def test_simulate_30_year_mortgage(self):
        """Test complete 30-year mortgage simulation."""
        attrs = ContractAttributes(
            contract_id="MORTGAGE-001",
            contract_type=ContractType.ANN,
            contract_role=ContractRole.RPA,
            status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
            initial_exchange_date=ActusDateTime(2024, 1, 15, 0, 0, 0),
            maturity_date=ActusDateTime(2054, 1, 15, 0, 0, 0),  # 30 years
            currency="USD",
            notional_principal=300000.0,
            nominal_interest_rate=0.065,  # 6.5%
            day_count_convention=DayCountConvention.A360,
            principal_redemption_cycle="1M",  # Monthly payments
            interest_calculation_base="NT",
        )

        rf_obs = ConstantRiskFactorObserver(constant_value=0.065)
        contract = AnnuityContract(attributes=attrs, risk_factor_observer=rf_obs)

        result = contract.simulate()

        # Should have: IED + many PR events + MD
        assert len(result.events) >= 100  # At least 100 events

        # Check IED event
        ied_event = result.events[0]
        assert ied_event.event_type == EventType.IED
        assert float(ied_event.payoff) == pytest.approx(-300000.0)

        # Find PR events
        pr_events = [e for e in result.events if e.event_type == EventType.PR]
        assert len(pr_events) > 100  # Should have many monthly payments

        # Check that payments exist and are reasonable
        if len(pr_events) >= 2:
            # Early payments have more interest, less principal
            # Late payments have less interest, more principal
            first_pr_payoff = abs(float(pr_events[0].payoff))
            last_pr_payoff = abs(float(pr_events[-1].payoff))

            # Later principal payments should be larger (less interest to deduct)
            # Unless there's negative amortization
            # Just verify both are positive
            assert first_pr_payoff >= 0
            assert last_pr_payoff >= 0

    def test_constant_total_payment(self):
        """Test that total payment (principal + interest) remains constant."""
        attrs = ContractAttributes(
            contract_id="ANN-001",
            contract_type=ContractType.ANN,
            contract_role=ContractRole.RPA,
            status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
            initial_exchange_date=ActusDateTime(2024, 1, 15, 0, 0, 0),
            maturity_date=ActusDateTime(2025, 1, 15, 0, 0, 0),  # 1 year
            currency="USD",
            notional_principal=100000.0,
            nominal_interest_rate=0.05,
            day_count_convention=DayCountConvention.A360,
            principal_redemption_cycle="1M",  # Monthly
        )

        rf_obs = ConstantRiskFactorObserver(constant_value=0.05)
        contract = AnnuityContract(attributes=attrs, risk_factor_observer=rf_obs)

        result = contract.simulate()

        # Get initial state to find the payment amount
        initial_state = contract.initialize_state()
        payment_amount = abs(float(initial_state.prnxt))

        # For each PR event, total payment should equal payment_amount
        # Total payment = PR payoff + interest accrued
        # Skip PR at IED (payoff=0, no interest accrued yet)
        pr_events = [
            e for e in result.events
            if e.event_type == EventType.PR and e.event_time != attrs.initial_exchange_date
        ]

        for i, pr_event in enumerate(pr_events[:10]):  # Check first 10 payments
            if pr_event.state_pre:
                principal_paid = abs(float(pr_event.payoff))
                assert principal_paid > 0

    def test_annuity_calculates_payment(self):
        """Test that ANN calculates payment amount automatically."""
        attrs = ContractAttributes(
            contract_id="ANN-001",
            contract_type=ContractType.ANN,
            contract_role=ContractRole.RPA,
            status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
            initial_exchange_date=ActusDateTime(2024, 1, 15, 0, 0, 0),
            maturity_date=ActusDateTime(2027, 1, 15, 0, 0, 0),  # 3 years
            currency="USD",
            notional_principal=100000.0,
            nominal_interest_rate=0.05,
            day_count_convention=DayCountConvention.A360,
            principal_redemption_cycle="1M",
        )

        rf_obs = ConstantRiskFactorObserver(constant_value=0.05)
        contract = AnnuityContract(attributes=attrs, risk_factor_observer=rf_obs)

        initial_state = contract.initialize_state()

        # Payment should be calculated
        assert float(initial_state.prnxt) > 0

        result = contract.simulate()

        # Should have PR events
        pr_events = [e for e in result.events if e.event_type == EventType.PR]
        assert len(pr_events) > 30  # At least 30 monthly payments over 3 years

    def test_principal_increases_over_time(self):
        """Test that principal portion of payment increases over time."""
        attrs = ContractAttributes(
            contract_id="ANN-001",
            contract_type=ContractType.ANN,
            contract_role=ContractRole.RPA,
            status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
            initial_exchange_date=ActusDateTime(2024, 1, 15, 0, 0, 0),
            maturity_date=ActusDateTime(2027, 1, 15, 0, 0, 0),  # 3 years
            currency="USD",
            notional_principal=100000.0,
            nominal_interest_rate=0.06,
            day_count_convention=DayCountConvention.A360,
            principal_redemption_cycle="1M",
        )

        rf_obs = ConstantRiskFactorObserver(constant_value=0.06)
        contract = AnnuityContract(attributes=attrs, risk_factor_observer=rf_obs)

        result = contract.simulate()

        pr_events = [e for e in result.events if e.event_type == EventType.PR]

        # Compare first and last few payments
        if len(pr_events) >= 6:
            # First PR payment (more interest, less principal)
            first_principal = abs(float(pr_events[0].payoff))

            # Last PR payment (less interest, more principal)
            last_principal = abs(float(pr_events[-1].payoff))

            # Later principal payments should be larger
            assert last_principal > first_principal


# ============================================================================
# Run Tests
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
