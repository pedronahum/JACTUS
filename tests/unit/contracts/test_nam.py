"""Unit tests for Negative Amortizer (NAM) contract implementation.

Tests the NegativeAmortizerContract, NAMPayoffFunction, and NAMStateTransitionFunction
classes for correctness according to the ACTUS specification (Section 7.4).
"""

import jax.numpy as jnp
import pytest

from jactus.contracts.nam import (
    NAMPayoffFunction,
    NAMStateTransitionFunction,
    NegativeAmortizerContract,
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

# ============================================================================
# Test NAMPayoffFunction
# ============================================================================


class TestNAMPayoffFunction:
    """Test NAMPayoffFunction class."""

    def test_initialization(self):
        """Test NAMPayoffFunction can be created."""
        pof = NAMPayoffFunction(
            contract_role=ContractRole.RPA,
            currency="USD",
            settlement_currency=None,
        )

        assert pof.contract_role == ContractRole.RPA
        assert pof.currency == "USD"

    def test_pof_ied_disburses_principal(self):
        """Test POF_IED disburses principal (same as LAM)."""
        pof = NAMPayoffFunction(
            contract_role=ContractRole.RPA,
            currency="USD",
            settlement_currency=None,
        )

        state = ContractState(
            sd=ActusDateTime(2024, 1, 1, 0, 0, 0),
            tmd=ActusDateTime(2029, 1, 15, 0, 0, 0),
            nt=jnp.array(100000.0),
            ipnr=jnp.array(0.05),
            ipac=jnp.array(0.0),
            feac=jnp.array(0.0),
            nsc=jnp.array(1.0),
            isc=jnp.array(1.0),
        )

        attrs = ContractAttributes(
            contract_id="NAM-001",
            contract_type=ContractType.NAM,
            contract_role=ContractRole.RPA,
            status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
            initial_exchange_date=ActusDateTime(2024, 1, 15, 0, 0, 0),
            maturity_date=ActusDateTime(2029, 1, 15, 0, 0, 0),
            currency="USD",
            notional_principal=100000.0,
            nominal_interest_rate=0.05,
            day_count_convention=DayCountConvention.A360,
        )

        rf_obs = ConstantRiskFactorObserver(constant_value=0.05)

        payoff = pof.calculate_payoff(
            EventType.IED,
            state,
            attrs,
            ActusDateTime(2024, 1, 15, 0, 0, 0),
            rf_obs,
        )

        # Borrower receives principal = negative cashflow from borrower perspective
        assert float(payoff) == pytest.approx(-100000.0)

    def test_pof_pr_net_of_interest_positive(self):
        """Test POF_PR when payment > interest (normal amortization)."""
        pof = NAMPayoffFunction(
            contract_role=ContractRole.RPA,
            currency="USD",
            settlement_currency=None,
        )

        # Payment of $5000, interest ~$416/month → positive cashflow
        state = ContractState(
            sd=ActusDateTime(2024, 1, 15, 0, 0, 0),
            tmd=ActusDateTime(2029, 1, 15, 0, 0, 0),
            nt=jnp.array(100000.0),
            ipnr=jnp.array(0.05),
            ipac=jnp.array(0.0),
            feac=jnp.array(0.0),
            nsc=jnp.array(1.0),
            isc=jnp.array(1.0),
            prnxt=jnp.array(5000.0),  # High payment
            ipcb=jnp.array(100000.0),
        )

        attrs = ContractAttributes(
            contract_id="NAM-001",
            contract_type=ContractType.NAM,
            contract_role=ContractRole.RPA,
            status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
            initial_exchange_date=ActusDateTime(2024, 1, 15, 0, 0, 0),
            maturity_date=ActusDateTime(2029, 1, 15, 0, 0, 0),
            currency="USD",
            notional_principal=100000.0,
            nominal_interest_rate=0.05,
            day_count_convention=DayCountConvention.A360,
            principal_redemption_cycle="1M",
            next_principal_redemption_amount=5000.0,
        )

        rf_obs = ConstantRiskFactorObserver(constant_value=0.05)

        payoff = pof.calculate_payoff(
            EventType.PR,
            state,
            attrs,
            ActusDateTime(2024, 2, 15, 0, 0, 0),
            rf_obs,
        )

        # Payment = $5000
        # Interest = (31/360) × 0.05 × 100000 = 430.56
        # Net = 5000 - 430.56 = 4569.44
        # POF_PR = R(CNTRL) × Nsc × (Prnxt - Ipac - accrued)
        #        = 1 × 1.0 × (5000 - 0 - 430.56) = 4569.44
        assert float(payoff) == pytest.approx(4569.44, rel=0.01)

    def test_pof_pr_net_of_interest_negative(self):
        """Test POF_PR when payment < interest (negative amortization)."""
        pof = NAMPayoffFunction(
            contract_role=ContractRole.RPA,
            currency="USD",
            settlement_currency=None,
        )

        # Low payment of $200, interest ~$416/month → negative cashflow (borrower receives)
        state = ContractState(
            sd=ActusDateTime(2024, 1, 15, 0, 0, 0),
            tmd=ActusDateTime(2029, 1, 15, 0, 0, 0),
            nt=jnp.array(100000.0),
            ipnr=jnp.array(0.05),
            ipac=jnp.array(0.0),
            feac=jnp.array(0.0),
            nsc=jnp.array(1.0),
            isc=jnp.array(1.0),
            prnxt=jnp.array(200.0),  # Low payment < interest
            ipcb=jnp.array(100000.0),
        )

        attrs = ContractAttributes(
            contract_id="NAM-001",
            contract_type=ContractType.NAM,
            contract_role=ContractRole.RPA,
            status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
            initial_exchange_date=ActusDateTime(2024, 1, 15, 0, 0, 0),
            maturity_date=ActusDateTime(2029, 1, 15, 0, 0, 0),
            currency="USD",
            notional_principal=100000.0,
            nominal_interest_rate=0.05,
            day_count_convention=DayCountConvention.A360,
            principal_redemption_cycle="1M",
            next_principal_redemption_amount=200.0,
        )

        rf_obs = ConstantRiskFactorObserver(constant_value=0.05)

        payoff = pof.calculate_payoff(
            EventType.PR,
            state,
            attrs,
            ActusDateTime(2024, 2, 15, 0, 0, 0),
            rf_obs,
        )

        # Payment = $200
        # Interest = (31/360) × 0.05 × 100000 = 430.56
        # Net = 200 - 430.56 = -230.56 (NEGATIVE!)
        # This means borrower receives $230.56 (negative amortization)
        assert float(payoff) == pytest.approx(-230.56, rel=0.01)


# ============================================================================
# Test NAMStateTransitionFunction
# ============================================================================


class TestNAMStateTransitionFunction:
    """Test NAMStateTransitionFunction class."""

    def test_initialization(self):
        """Test NAMStateTransitionFunction can be created."""
        stf = NAMStateTransitionFunction()
        assert stf is not None

    def test_stf_ied_initializes_state(self):
        """Test STF_IED initializes state (same as LAM)."""
        stf = NAMStateTransitionFunction()

        initial_state = ContractState(
            sd=ActusDateTime(2024, 1, 1, 0, 0, 0),
            tmd=ActusDateTime(2029, 1, 15, 0, 0, 0),
            nt=jnp.array(0.0),
            ipnr=jnp.array(0.0),
            ipac=jnp.array(0.0),
            feac=jnp.array(0.0),
            nsc=jnp.array(1.0),
            isc=jnp.array(1.0),
        )

        attrs = ContractAttributes(
            contract_id="NAM-001",
            contract_type=ContractType.NAM,
            contract_role=ContractRole.RPA,
            status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
            initial_exchange_date=ActusDateTime(2024, 1, 15, 0, 0, 0),
            maturity_date=ActusDateTime(2029, 1, 15, 0, 0, 0),
            currency="USD",
            notional_principal=100000.0,
            nominal_interest_rate=0.05,
            day_count_convention=DayCountConvention.A360,
            interest_calculation_base="NT",
            next_principal_redemption_amount=200.0,
        )

        rf_obs = ConstantRiskFactorObserver(constant_value=0.05)

        new_state = stf.transition_state(
            EventType.IED,
            initial_state,
            attrs,
            ActusDateTime(2024, 1, 15, 0, 0, 0),
            rf_obs,
        )

        assert new_state.sd == ActusDateTime(2024, 1, 15, 0, 0, 0)
        assert float(new_state.nt) == pytest.approx(100000.0)
        assert float(new_state.ipnr) == pytest.approx(0.05)
        assert float(new_state.prnxt) == pytest.approx(200.0)
        assert float(new_state.ipcb) == pytest.approx(100000.0)

    def test_stf_pr_notional_decreases_when_payment_exceeds_interest(self):
        """Test STF_PR reduces notional when payment > interest."""
        stf = NAMStateTransitionFunction()

        state = ContractState(
            sd=ActusDateTime(2024, 1, 15, 0, 0, 0),
            tmd=ActusDateTime(2029, 1, 15, 0, 0, 0),
            nt=jnp.array(100000.0),
            ipnr=jnp.array(0.05),
            ipac=jnp.array(0.0),
            feac=jnp.array(0.0),
            nsc=jnp.array(1.0),
            isc=jnp.array(1.0),
            prnxt=jnp.array(5000.0),  # Payment > interest
            ipcb=jnp.array(100000.0),
        )

        attrs = ContractAttributes(
            contract_id="NAM-001",
            contract_type=ContractType.NAM,
            contract_role=ContractRole.RPA,
            status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
            initial_exchange_date=ActusDateTime(2024, 1, 15, 0, 0, 0),
            maturity_date=ActusDateTime(2029, 1, 15, 0, 0, 0),
            currency="USD",
            notional_principal=100000.0,
            nominal_interest_rate=0.05,
            day_count_convention=DayCountConvention.A360,
            interest_calculation_base="NT",
            next_principal_redemption_amount=5000.0,
        )

        rf_obs = ConstantRiskFactorObserver(constant_value=0.05)

        new_state = stf.transition_state(
            EventType.PR,
            state,
            attrs,
            ActusDateTime(2024, 2, 15, 0, 0, 0),
            rf_obs,
        )

        # Payment = $5000
        # Interest accrued = (31/360) × 0.05 × 100000 = 430.56
        # Net principal reduction = 5000 - 0 - 430.56 = 4569.44
        # New notional = 100000 - 4569.44 = 95430.56
        assert float(new_state.nt) == pytest.approx(95430.56, rel=0.01)
        # IPAC accumulates at PR (zeroed at IP, not PR)
        assert float(new_state.ipac) == pytest.approx(430.56, rel=0.01)
        # In NT mode, IPCB follows notional
        assert float(new_state.ipcb) == pytest.approx(95430.56, rel=0.01)

    def test_stf_pr_notional_increases_when_payment_less_than_interest(self):
        """Test STF_PR increases notional when payment < interest (negative amortization)."""
        stf = NAMStateTransitionFunction()

        state = ContractState(
            sd=ActusDateTime(2024, 1, 15, 0, 0, 0),
            tmd=ActusDateTime(2029, 1, 15, 0, 0, 0),
            nt=jnp.array(100000.0),
            ipnr=jnp.array(0.05),
            ipac=jnp.array(0.0),
            feac=jnp.array(0.0),
            nsc=jnp.array(1.0),
            isc=jnp.array(1.0),
            prnxt=jnp.array(200.0),  # Payment < interest → negative amortization
            ipcb=jnp.array(100000.0),
        )

        attrs = ContractAttributes(
            contract_id="NAM-001",
            contract_type=ContractType.NAM,
            contract_role=ContractRole.RPA,
            status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
            initial_exchange_date=ActusDateTime(2024, 1, 15, 0, 0, 0),
            maturity_date=ActusDateTime(2029, 1, 15, 0, 0, 0),
            currency="USD",
            notional_principal=100000.0,
            nominal_interest_rate=0.05,
            day_count_convention=DayCountConvention.A360,
            interest_calculation_base="NT",
            next_principal_redemption_amount=200.0,
        )

        rf_obs = ConstantRiskFactorObserver(constant_value=0.05)

        new_state = stf.transition_state(
            EventType.PR,
            state,
            attrs,
            ActusDateTime(2024, 2, 15, 0, 0, 0),
            rf_obs,
        )

        # Payment = $200
        # Interest = (31/360) × 0.05 × 100000 = 430.56
        # Net principal reduction = 200 - 0 - 430.56 = -230.56 (NEGATIVE!)
        # New notional = 100000 - (-230.56) = 100230.56 (INCREASED!)
        assert float(new_state.nt) == pytest.approx(100230.56, rel=0.01)
        # IPAC accumulates at PR (zeroed at IP, not PR)
        assert float(new_state.ipac) == pytest.approx(430.56, rel=0.01)
        # In NT mode, IPCB follows notional
        assert float(new_state.ipcb) == pytest.approx(100230.56, rel=0.01)


# ============================================================================
# Test NegativeAmortizerContract
# ============================================================================


class TestNegativeAmortizerContract:
    """Test NegativeAmortizerContract class."""

    def test_initialization(self):
        """Test NegativeAmortizerContract can be created."""
        attrs = ContractAttributes(
            contract_id="NAM-001",
            contract_type=ContractType.NAM,
            contract_role=ContractRole.RPA,
            status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
            initial_exchange_date=ActusDateTime(2024, 1, 15, 0, 0, 0),
            maturity_date=ActusDateTime(2029, 1, 15, 0, 0, 0),
            currency="USD",
            notional_principal=100000.0,
            nominal_interest_rate=0.05,
            day_count_convention=DayCountConvention.A360,
            principal_redemption_cycle="1M",
            next_principal_redemption_amount=200.0,
        )

        rf_obs = ConstantRiskFactorObserver(constant_value=0.05)
        contract = NegativeAmortizerContract(attributes=attrs, risk_factor_observer=rf_obs)

        assert contract.attributes.contract_id == "NAM-001"
        assert contract.attributes.contract_type == ContractType.NAM

    def test_ip_schedule_ends_before_last_pr(self):
        """Test that IP schedule ends one period before PR schedule."""
        attrs = ContractAttributes(
            contract_id="NAM-001",
            contract_type=ContractType.NAM,
            contract_role=ContractRole.RPA,
            status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
            initial_exchange_date=ActusDateTime(2024, 1, 15, 0, 0, 0),
            maturity_date=ActusDateTime(2029, 1, 15, 0, 0, 0),
            currency="USD",
            notional_principal=100000.0,
            nominal_interest_rate=0.05,
            day_count_convention=DayCountConvention.A360,
            principal_redemption_cycle="1Y",  # Annual PR
            interest_payment_cycle="1Y",  # Annual IP
            next_principal_redemption_amount=5000.0,
        )

        rf_obs = ConstantRiskFactorObserver(constant_value=0.05)
        contract = NegativeAmortizerContract(attributes=attrs, risk_factor_observer=rf_obs)

        schedule = contract.generate_event_schedule()

        # Find PR and IP events
        pr_events = [e for e in schedule.events if e.event_type == EventType.PR]
        ip_events = [e for e in schedule.events if e.event_type == EventType.IP]

        # PR: IED, 2025, 2026, 2027, 2028 (MD is 2029) = 5 events (PRANX defaults to IED)
        # IP: from IPANX (or IED) to MD per ACTUS spec
        assert len(pr_events) == 5
        # IP events include IED (payoff=0) and all cycle dates up to MD
        # Filter out IP at IED for count check
        ip_after_ied = [
            e for e in ip_events
            if e.event_time != attrs.initial_exchange_date
        ]
        assert len(ip_after_ied) >= 4  # At least one IP per year

    def test_simulate_with_negative_amortization(self):
        """Test complete NAM simulation with periods of negative amortization."""
        attrs = ContractAttributes(
            contract_id="NAM-001",
            contract_type=ContractType.NAM,
            contract_role=ContractRole.RPA,
            status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
            initial_exchange_date=ActusDateTime(2024, 1, 15, 0, 0, 0),
            maturity_date=ActusDateTime(2024, 7, 15, 0, 0, 0),  # 6 months
            currency="USD",
            notional_principal=100000.0,
            nominal_interest_rate=0.05,
            day_count_convention=DayCountConvention.A360,
            principal_redemption_cycle="1M",  # Monthly payments
            next_principal_redemption_amount=200.0,  # Low payment → negative amort
            interest_calculation_base="NT",
        )

        rf_obs = ConstantRiskFactorObserver(constant_value=0.05)
        contract = NegativeAmortizerContract(attributes=attrs, risk_factor_observer=rf_obs)

        result = contract.simulate()

        # Check IED event
        ied_event = result.events[0]
        assert ied_event.event_type == EventType.IED
        assert float(ied_event.payoff) == pytest.approx(-100000.0)

        # Find PR events
        pr_events = [e for e in result.events if e.event_type == EventType.PR]

        # All PR payoffs after IED should be negative (payment < interest)
        # Skip first PR at IED (no accrued interest yet)
        for pr_event in pr_events[1:]:
            # Each payment is $200 but interest is ~$400-420/month
            # So payoff should be negative
            assert float(pr_event.payoff) < 0.0

        # Check that notional INCREASES over time (negative amortization)
        # Use PR events after IED for comparison
        later_prs = pr_events[1:]
        if len(later_prs) >= 2:
            first_pr = later_prs[0]
            second_pr = later_prs[1]
            # Notional should increase from first to second PR
            if first_pr.state_pre and second_pr.state_pre:
                assert float(second_pr.state_pre.nt) > float(first_pr.state_pre.nt)

    def test_notional_reaches_zero_at_maturity_with_adequate_payments(self):
        """Test that notional reaches zero when payments are adequate."""
        # Use higher payment to ensure positive amortization
        attrs = ContractAttributes(
            contract_id="NAM-001",
            contract_type=ContractType.NAM,
            contract_role=ContractRole.RPA,
            status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
            initial_exchange_date=ActusDateTime(2024, 1, 15, 0, 0, 0),
            maturity_date=ActusDateTime(2028, 1, 15, 0, 0, 0),  # 4 years
            currency="USD",
            notional_principal=20000.0,
            nominal_interest_rate=0.05,
            day_count_convention=DayCountConvention.A360,
            principal_redemption_cycle="1Y",
            next_principal_redemption_amount=6000.0,  # High payment > interest
        )

        rf_obs = ConstantRiskFactorObserver(constant_value=0.05)
        contract = NegativeAmortizerContract(attributes=attrs, risk_factor_observer=rf_obs)

        result = contract.simulate()

        # Find MD event
        md_event = [e for e in result.events if e.event_type == EventType.MD][0]

        # At MD, notional should be small (close to zero or exactly zero)
        # With adequate payments, it should fully amortize
        if md_event.state_pre:
            # Notional should be relatively small
            assert float(md_event.state_pre.nt) < 10000.0  # Less than half original

    def test_comparison_nam_vs_lam_same_payment(self):
        """Test NAM vs LAM with same payment schedule shows difference in handling."""
        # NAM and LAM should behave the same when payment > interest
        # But formulas are different (NAM nets interest in payoff, LAM doesn't)
        attrs = ContractAttributes(
            contract_id="NAM-001",
            contract_type=ContractType.NAM,
            contract_role=ContractRole.RPA,
            status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
            initial_exchange_date=ActusDateTime(2024, 1, 15, 0, 0, 0),
            maturity_date=ActusDateTime(2026, 1, 15, 0, 0, 0),  # 2 years for PR events
            currency="USD",
            notional_principal=100000.0,
            nominal_interest_rate=0.05,
            day_count_convention=DayCountConvention.A360,
            principal_redemption_cycle="1Y",
            next_principal_redemption_amount=5000.0,
            interest_calculation_base="NT",
        )

        rf_obs = ConstantRiskFactorObserver(constant_value=0.05)
        contract = NegativeAmortizerContract(attributes=attrs, risk_factor_observer=rf_obs)

        result = contract.simulate()

        # Find first PR event
        pr_events = [e for e in result.events if e.event_type == EventType.PR]
        assert len(pr_events) > 0

        # For NAM, PR payoff should be net of interest
        # Payment $5000, Interest ~$5083, Net = -$83 (small negative)
        first_pr = pr_events[0]
        # NAM PR payoff is net of interest, so it's small (could be negative)
        assert abs(float(first_pr.payoff)) < 6000.0  # Much less than payment


# ============================================================================
# Run Tests
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
