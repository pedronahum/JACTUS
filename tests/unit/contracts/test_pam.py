"""Unit tests for Principal At Maturity (PAM) contract implementation.

Tests the PrincipalAtMaturityContract, PAMPayoffFunction, and PAMStateTransitionFunction
classes for correctness according to the ACTUS specification.
"""

import jax.numpy as jnp
import pytest
from pydantic_core import ValidationError

from jactus.contracts.pam import (
    PAMPayoffFunction,
    PAMStateTransitionFunction,
    PrincipalAtMaturityContract,
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
# Test PAMPayoffFunction
# ============================================================================


class TestPAMPayoffFunction:
    """Test PAMPayoffFunction class."""

    def test_initialization(self):
        """Test PAMPayoffFunction can be created."""
        pof = PAMPayoffFunction(
            contract_role=ContractRole.RPA,
            currency="USD",
            settlement_currency=None,
        )

        assert pof.contract_role == ContractRole.RPA
        assert pof.currency == "USD"

    def test_pof_ad_returns_zero(self):
        """Test POF_AD_PAM returns zero (no cashflow at analysis)."""
        pof = PAMPayoffFunction(
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
            contract_id="PAM-001",
            contract_type=ContractType.PAM,
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
            EventType.AD,
            state,
            attrs,
            ActusDateTime(2024, 1, 1, 0, 0, 0),
            rf_obs,
        )

        assert float(payoff) == 0.0

    def test_pof_ied_disburses_principal(self):
        """Test POF_IED_PAM disburses principal (negative cashflow)."""
        pof = PAMPayoffFunction(
            contract_role=ContractRole.RPA,
            currency="USD",
            settlement_currency=None,
        )

        state = ContractState(
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
            contract_id="PAM-001",
            contract_type=ContractType.PAM,
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

        # IED should disburse -100000 (negative = outflow to borrower)
        assert float(payoff) == -100000.0

    def test_pof_ied_with_premium_discount(self):
        """Test POF_IED_PAM with premium/discount."""
        pof = PAMPayoffFunction(
            contract_role=ContractRole.RPA,
            currency="USD",
            settlement_currency=None,
        )

        state = ContractState(
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
            contract_id="PAM-001",
            contract_type=ContractType.PAM,
            contract_role=ContractRole.RPA,
            status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
            initial_exchange_date=ActusDateTime(2024, 1, 15, 0, 0, 0),
            maturity_date=ActusDateTime(2029, 1, 15, 0, 0, 0),
            currency="USD",
            notional_principal=100000.0,
            nominal_interest_rate=0.05,
            premium_discount_at_ied=1000.0,  # Premium
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

        # IED should disburse -(100000 + 1000) = -101000
        assert float(payoff) == -101000.0

    def test_pof_md_returns_principal(self):
        """Test POF_MD_PAM returns principal at maturity."""
        pof = PAMPayoffFunction(
            contract_role=ContractRole.RPA,
            currency="USD",
            settlement_currency=None,
        )

        state = ContractState(
            sd=ActusDateTime(2029, 1, 1, 0, 0, 0),
            tmd=ActusDateTime(2029, 1, 15, 0, 0, 0),
            nt=jnp.array(100000.0),
            ipnr=jnp.array(0.05),
            ipac=jnp.array(0.0),
            feac=jnp.array(0.0),
            nsc=jnp.array(1.0),
            isc=jnp.array(1.0),
        )

        attrs = ContractAttributes(
            contract_id="PAM-001",
            contract_type=ContractType.PAM,
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
            EventType.MD,
            state,
            attrs,
            ActusDateTime(2029, 1, 15, 0, 0, 0),
            rf_obs,
        )

        # MD should return principal: nsc * nt = 1.0 * 100000 = 100000
        assert float(payoff) == 100000.0

    def test_pof_md_with_accrued_interest(self):
        """Test POF_MD_PAM includes accrued interest."""
        pof = PAMPayoffFunction(
            contract_role=ContractRole.RPA,
            currency="USD",
            settlement_currency=None,
        )

        state = ContractState(
            sd=ActusDateTime(2029, 1, 1, 0, 0, 0),
            tmd=ActusDateTime(2029, 1, 15, 0, 0, 0),
            nt=jnp.array(100000.0),
            ipnr=jnp.array(0.05),
            ipac=jnp.array(500.0),  # Accrued interest
            feac=jnp.array(0.0),
            nsc=jnp.array(1.0),
            isc=jnp.array(1.0),
        )

        attrs = ContractAttributes(
            contract_id="PAM-001",
            contract_type=ContractType.PAM,
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
            EventType.MD,
            state,
            attrs,
            ActusDateTime(2029, 1, 15, 0, 0, 0),
            rf_obs,
        )

        # MD should return: nsc * nt + isc * ipac = 1.0 * 100000 + 1.0 * 500 = 100500
        assert float(payoff) == 100500.0

    def test_pof_ip_calculates_interest_payment(self):
        """Test POF_IP_PAM calculates interest payment correctly."""
        pof = PAMPayoffFunction(
            contract_role=ContractRole.RPA,
            currency="USD",
            settlement_currency=None,
        )

        # State after 1 year, before IP event
        state = ContractState(
            sd=ActusDateTime(2024, 1, 15, 0, 0, 0),  # Last status date
            tmd=ActusDateTime(2029, 1, 15, 0, 0, 0),
            nt=jnp.array(100000.0),
            ipnr=jnp.array(0.05),
            ipac=jnp.array(0.0),  # No prior accrual
            feac=jnp.array(0.0),
            nsc=jnp.array(1.0),
            isc=jnp.array(1.0),
        )

        attrs = ContractAttributes(
            contract_id="PAM-001",
            contract_type=ContractType.PAM,
            contract_role=ContractRole.RPA,
            status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
            initial_exchange_date=ActusDateTime(2024, 1, 15, 0, 0, 0),
            maturity_date=ActusDateTime(2029, 1, 15, 0, 0, 0),
            currency="USD",
            notional_principal=100000.0,
            nominal_interest_rate=0.05,
            day_count_convention=DayCountConvention.A360,
            interest_payment_cycle="1Y",
        )

        rf_obs = ConstantRiskFactorObserver(constant_value=0.05)

        # IP event at 2025-01-15 (1 year after IED)
        payoff = pof.calculate_payoff(
            EventType.IP,
            state,
            attrs,
            ActusDateTime(2025, 1, 15, 0, 0, 0),
            rf_obs,
        )

        # Interest payment: isc * (ipac + Y(sd, t) * ipnr * nt)
        # Y(2024-01-15, 2025-01-15) with A360 = 366/360 = 1.01666... (2024 is leap year)
        # IP = 1.0 * (0.0 + 1.01666... * 0.05 * 100000) ≈ 5083.33
        expected = 5083.33
        assert abs(float(payoff) - expected) < 1.0  # Allow small rounding

    def test_pof_ipci_returns_zero(self):
        """Test POF_IPCI_PAM returns zero (capitalization has no cashflow)."""
        pof = PAMPayoffFunction(
            contract_role=ContractRole.RPA,
            currency="USD",
            settlement_currency=None,
        )

        state = ContractState(
            sd=ActusDateTime(2024, 1, 15, 0, 0, 0),
            tmd=ActusDateTime(2029, 1, 15, 0, 0, 0),
            nt=jnp.array(100000.0),
            ipnr=jnp.array(0.05),
            ipac=jnp.array(5000.0),
            feac=jnp.array(0.0),
            nsc=jnp.array(1.0),
            isc=jnp.array(1.0),
        )

        attrs = ContractAttributes(
            contract_id="PAM-001",
            contract_type=ContractType.PAM,
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
            EventType.IPCI,
            state,
            attrs,
            ActusDateTime(2025, 1, 15, 0, 0, 0),
            rf_obs,
        )

        # IPCI has no cashflow (interest added to principal)
        assert float(payoff) == 0.0


# ============================================================================
# Test PAMStateTransitionFunction
# ============================================================================


class TestPAMStateTransitionFunction:
    """Test PAMStateTransitionFunction class."""

    def test_initialization(self):
        """Test PAMStateTransitionFunction can be created."""
        stf = PAMStateTransitionFunction()
        assert stf is not None

    def test_stf_ad_accrues_interest(self):
        """Test STF_AD_PAM accrues interest."""
        stf = PAMStateTransitionFunction()

        state_pre = ContractState(
            sd=ActusDateTime(2024, 1, 15, 0, 0, 0),  # IED
            tmd=ActusDateTime(2029, 1, 15, 0, 0, 0),
            nt=jnp.array(100000.0),
            ipnr=jnp.array(0.05),
            ipac=jnp.array(0.0),
            feac=jnp.array(0.0),
            nsc=jnp.array(1.0),
            isc=jnp.array(1.0),
        )

        attrs = ContractAttributes(
            contract_id="PAM-001",
            contract_type=ContractType.PAM,
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

        # AD event 6 months after IED
        state_post = stf.transition_state(
            EventType.AD,
            state_pre,
            attrs,
            ActusDateTime(2024, 7, 15, 0, 0, 0),
            rf_obs,
        )

        # Status date should be updated
        assert state_post.sd == ActusDateTime(2024, 7, 15, 0, 0, 0)

        # IPAC should have accrued: ipac + Y(sd, t) * ipnr * nt
        # Y(2024-01-15, 2024-07-15) with A360 = 182/360 = 0.5055... (actual days)
        # ipac = 0.0 + 0.5055 * 0.05 * 100000 ≈ 2527.78
        expected_ipac = 2527.78
        assert abs(float(state_post.ipac) - expected_ipac) < 1.0

        # Notional should be unchanged
        assert float(state_post.nt) == 100000.0

    def test_stf_ied_initializes_state(self):
        """Test STF_IED_PAM initializes notional and rate."""
        stf = PAMStateTransitionFunction()

        state_pre = ContractState(
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
            contract_id="PAM-001",
            contract_type=ContractType.PAM,
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

        state_post = stf.transition_state(
            EventType.IED,
            state_pre,
            attrs,
            ActusDateTime(2024, 1, 15, 0, 0, 0),
            rf_obs,
        )

        # Status date should be IED
        assert state_post.sd == ActusDateTime(2024, 1, 15, 0, 0, 0)

        # Notional should be set: R(CNTRL) * NT = +1 * 100000 = 100000
        assert float(state_post.nt) == 100000.0

        # Interest rate should be set
        assert abs(float(state_post.ipnr) - 0.05) < 1e-6

        # IPAC should be zero at IED
        assert float(state_post.ipac) == 0.0

    def test_stf_ied_applies_role_sign(self):
        """Test STF_IED_PAM applies role sign to notional."""
        stf = PAMStateTransitionFunction()

        state_pre = ContractState(
            sd=ActusDateTime(2024, 1, 1, 0, 0, 0),
            tmd=ActusDateTime(2029, 1, 15, 0, 0, 0),
            nt=jnp.array(0.0),
            ipnr=jnp.array(0.0),
            ipac=jnp.array(0.0),
            feac=jnp.array(0.0),
            nsc=jnp.array(1.0),
            isc=jnp.array(1.0),
        )

        # Test with RPL role (lender perspective = negative notional)
        attrs_rpl = ContractAttributes(
            contract_id="PAM-001",
            contract_type=ContractType.PAM,
            contract_role=ContractRole.RPL,  # Lender
            status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
            initial_exchange_date=ActusDateTime(2024, 1, 15, 0, 0, 0),
            maturity_date=ActusDateTime(2029, 1, 15, 0, 0, 0),
            currency="USD",
            notional_principal=100000.0,
            nominal_interest_rate=0.05,
            day_count_convention=DayCountConvention.A360,
        )

        rf_obs = ConstantRiskFactorObserver(constant_value=0.05)

        state_post_rpl = stf.transition_state(
            EventType.IED,
            state_pre,
            attrs_rpl,
            ActusDateTime(2024, 1, 15, 0, 0, 0),
            rf_obs,
        )

        # RPL should have negative notional
        assert float(state_post_rpl.nt) == -100000.0

        # Test with RPA role (payer perspective = positive notional)
        attrs_rpa = ContractAttributes(
            contract_id="PAM-001",
            contract_type=ContractType.PAM,
            contract_role=ContractRole.RPA,  # Payer/Borrower
            status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
            initial_exchange_date=ActusDateTime(2024, 1, 15, 0, 0, 0),
            maturity_date=ActusDateTime(2029, 1, 15, 0, 0, 0),
            currency="USD",
            notional_principal=100000.0,
            nominal_interest_rate=0.05,
            day_count_convention=DayCountConvention.A360,
        )

        state_post_rpa = stf.transition_state(
            EventType.IED,
            state_pre,
            attrs_rpa,
            ActusDateTime(2024, 1, 15, 0, 0, 0),
            rf_obs,
        )

        # RPA should have positive notional
        assert float(state_post_rpa.nt) == 100000.0

    def test_stf_md_zeros_state(self):
        """Test STF_MD_PAM zeros out notional and accruals."""
        stf = PAMStateTransitionFunction()

        state_pre = ContractState(
            sd=ActusDateTime(2029, 1, 1, 0, 0, 0),
            tmd=ActusDateTime(2029, 1, 15, 0, 0, 0),
            nt=jnp.array(100000.0),
            ipnr=jnp.array(0.05),
            ipac=jnp.array(500.0),
            feac=jnp.array(0.0),
            nsc=jnp.array(1.0),
            isc=jnp.array(1.0),
        )

        attrs = ContractAttributes(
            contract_id="PAM-001",
            contract_type=ContractType.PAM,
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

        state_post = stf.transition_state(
            EventType.MD,
            state_pre,
            attrs,
            ActusDateTime(2029, 1, 15, 0, 0, 0),
            rf_obs,
        )

        # Status date should be MD
        assert state_post.sd == ActusDateTime(2029, 1, 15, 0, 0, 0)

        # Notional and accruals should be zero after maturity
        assert float(state_post.nt) == 0.0
        assert float(state_post.ipac) == 0.0
        assert float(state_post.feac) == 0.0
        # Rate is preserved per ACTUS spec
        assert float(state_post.ipnr) == pytest.approx(0.05)

    def test_stf_ip_resets_accrued_interest(self):
        """Test STF_IP_PAM resets accrued interest after payment."""
        stf = PAMStateTransitionFunction()

        state_pre = ContractState(
            sd=ActusDateTime(2024, 1, 15, 0, 0, 0),
            tmd=ActusDateTime(2029, 1, 15, 0, 0, 0),
            nt=jnp.array(100000.0),
            ipnr=jnp.array(0.05),
            ipac=jnp.array(5000.0),  # Accrued interest
            feac=jnp.array(0.0),
            nsc=jnp.array(1.0),
            isc=jnp.array(1.0),
        )

        attrs = ContractAttributes(
            contract_id="PAM-001",
            contract_type=ContractType.PAM,
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

        state_post = stf.transition_state(
            EventType.IP,
            state_pre,
            attrs,
            ActusDateTime(2025, 1, 15, 0, 0, 0),
            rf_obs,
        )

        # Status date should be updated
        assert state_post.sd == ActusDateTime(2025, 1, 15, 0, 0, 0)

        # IPAC should be reset to zero after payment
        assert float(state_post.ipac) == 0.0

        # Notional should be unchanged
        assert float(state_post.nt) == 100000.0

        # Rate should be unchanged
        assert abs(float(state_post.ipnr) - 0.05) < 1e-6

    def test_stf_ipci_capitalizes_interest(self):
        """Test STF_IPCI_PAM adds accrued interest to notional."""
        stf = PAMStateTransitionFunction()

        state_pre = ContractState(
            sd=ActusDateTime(2024, 1, 15, 0, 0, 0),
            tmd=ActusDateTime(2029, 1, 15, 0, 0, 0),
            nt=jnp.array(100000.0),
            ipnr=jnp.array(0.05),
            ipac=jnp.array(0.0),  # No prior accrual
            feac=jnp.array(0.0),
            nsc=jnp.array(1.0),
            isc=jnp.array(1.0),
        )

        attrs = ContractAttributes(
            contract_id="PAM-001",
            contract_type=ContractType.PAM,
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

        # IPCI event 1 year after IED
        state_post = stf.transition_state(
            EventType.IPCI,
            state_pre,
            attrs,
            ActusDateTime(2025, 1, 15, 0, 0, 0),
            rf_obs,
        )

        # Status date should be updated
        assert state_post.sd == ActusDateTime(2025, 1, 15, 0, 0, 0)

        # IPAC should be reset to zero (capitalized)
        assert float(state_post.ipac) == 0.0

        # Notional should increase by accrued interest
        # Y(2024-01-15, 2025-01-15) with A360 = 366/360 = 1.01666... (2024 is leap year)
        # ipac = 0.0 + 1.01666 * 0.05 * 100000 ≈ 5083.33
        # new nt = 100000 + 5083.33 ≈ 105083.33
        expected_nt = 105083.33
        assert abs(float(state_post.nt) - expected_nt) < 1.0

        # Rate should be unchanged
        assert abs(float(state_post.ipnr) - 0.05) < 1e-6


# ============================================================================
# Test PrincipalAtMaturityContract
# ============================================================================


class TestPrincipalAtMaturityContract:
    """Test PrincipalAtMaturityContract class."""

    def test_initialization_success(self):
        """Test PAM contract can be created with valid attributes."""
        attrs = ContractAttributes(
            contract_id="PAM-001",
            contract_type=ContractType.PAM,
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

        contract = PrincipalAtMaturityContract(
            attributes=attrs,
            risk_factor_observer=rf_obs,
        )

        assert contract.attributes.contract_id == "PAM-001"
        assert contract.attributes.contract_type == ContractType.PAM

    def test_initialization_requires_contract_type(self):
        """Test PAM contract requires PAM contract type."""
        with pytest.raises(ValueError) as exc_info:
            attrs = ContractAttributes(
                contract_id="PAM-001",
                contract_type=ContractType.CSH,  # Wrong type
                contract_role=ContractRole.RPA,
                status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
                initial_exchange_date=ActusDateTime(2024, 1, 15, 0, 0, 0),
                maturity_date=ActusDateTime(2029, 1, 15, 0, 0, 0),
                currency="USD",
                notional_principal=100000.0,
            )

            rf_obs = ConstantRiskFactorObserver(constant_value=0.0)

            PrincipalAtMaturityContract(
                attributes=attrs,
                risk_factor_observer=rf_obs,
            )

        assert "contract type must be pam" in str(exc_info.value).lower()

    def test_initialization_requires_ied(self):
        """Test PAM contract requires initial_exchange_date."""
        with pytest.raises(ValueError) as exc_info:
            attrs = ContractAttributes(
                contract_id="PAM-001",
                contract_type=ContractType.PAM,
                contract_role=ContractRole.RPA,
                status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
                initial_exchange_date=None,  # Missing
                maturity_date=ActusDateTime(2029, 1, 15, 0, 0, 0),
                currency="USD",
                notional_principal=100000.0,
            )

            rf_obs = ConstantRiskFactorObserver(constant_value=0.0)

            PrincipalAtMaturityContract(
                attributes=attrs,
                risk_factor_observer=rf_obs,
            )

        assert "initial_exchange_date" in str(exc_info.value).lower()

    def test_initialization_requires_maturity_date(self):
        """Test PAM contract requires maturity_date."""
        with pytest.raises(ValueError) as exc_info:
            attrs = ContractAttributes(
                contract_id="PAM-001",
                contract_type=ContractType.PAM,
                contract_role=ContractRole.RPA,
                status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
                initial_exchange_date=ActusDateTime(2024, 1, 15, 0, 0, 0),
                maturity_date=None,  # Missing
                currency="USD",
                notional_principal=100000.0,
            )

            rf_obs = ConstantRiskFactorObserver(constant_value=0.0)

            PrincipalAtMaturityContract(
                attributes=attrs,
                risk_factor_observer=rf_obs,
            )

        assert "maturity_date" in str(exc_info.value).lower()

    def test_initialization_requires_notional(self):
        """Test PAM contract requires notional_principal."""
        with pytest.raises(ValueError) as exc_info:
            attrs = ContractAttributes(
                contract_id="PAM-001",
                contract_type=ContractType.PAM,
                contract_role=ContractRole.RPA,
                status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
                initial_exchange_date=ActusDateTime(2024, 1, 15, 0, 0, 0),
                maturity_date=ActusDateTime(2029, 1, 15, 0, 0, 0),
                currency="USD",
                notional_principal=None,  # Missing
            )

            rf_obs = ConstantRiskFactorObserver(constant_value=0.0)

            PrincipalAtMaturityContract(
                attributes=attrs,
                risk_factor_observer=rf_obs,
            )

        assert "notional_principal" in str(exc_info.value).lower()

    def test_initialization_validates_date_ordering(self):
        """Test PAM contract validates MD > IED. IED < SD is allowed (ACTUS spec)."""
        # IED < status_date is allowed per ACTUS spec (contract already started)
        attrs = ContractAttributes(
            contract_id="PAM-001",
            contract_type=ContractType.PAM,
            contract_role=ContractRole.RPA,
            status_date=ActusDateTime(2024, 1, 15, 0, 0, 0),
            initial_exchange_date=ActusDateTime(2024, 1, 1, 0, 0, 0),  # Before status
            maturity_date=ActusDateTime(2029, 1, 15, 0, 0, 0),
            currency="USD",
            notional_principal=100000.0,
        )
        assert attrs is not None

        # Test MD <= IED - Pydantic catches this during attribute creation
        with pytest.raises((ValueError, ValidationError)) as exc_info:
            attrs = ContractAttributes(
                contract_id="PAM-001",
                contract_type=ContractType.PAM,
                contract_role=ContractRole.RPA,
                status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
                initial_exchange_date=ActusDateTime(2029, 1, 15, 0, 0, 0),
                maturity_date=ActusDateTime(2024, 1, 15, 0, 0, 0),  # Before IED
                currency="USD",
                notional_principal=100000.0,
            )

        # Should contain information about maturity and initial exchange
        error_msg2 = str(exc_info.value).lower()
        assert ("maturity" in error_msg2 or "md" in error_msg2) and (
            "initial" in error_msg2 or "ied" in error_msg2
        )

    def test_initialize_state_before_ied(self):
        """Test PAM state initialization before IED has zero notional."""
        attrs = ContractAttributes(
            contract_id="PAM-001",
            contract_type=ContractType.PAM,
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

        contract = PrincipalAtMaturityContract(
            attributes=attrs,
            risk_factor_observer=rf_obs,
        )

        state = contract.initialize_state()

        # Before IED, notional should be zero
        assert float(state.nt) == 0.0

        # Before IED, rate should be zero
        assert float(state.ipnr) == 0.0

        # TMD should be set to maturity date
        assert state.tmd == ActusDateTime(2029, 1, 15, 0, 0, 0)

        # Status date should be status_date
        assert state.sd == ActusDateTime(2024, 1, 1, 0, 0, 0)

    def test_generate_event_schedule_basic(self):
        """Test PAM event schedule generation with basic loan."""
        attrs = ContractAttributes(
            contract_id="PAM-001",
            contract_type=ContractType.PAM,
            contract_role=ContractRole.RPA,
            status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
            initial_exchange_date=ActusDateTime(2024, 1, 15, 0, 0, 0),
            maturity_date=ActusDateTime(2027, 1, 15, 0, 0, 0),
            currency="USD",
            notional_principal=100000.0,
            nominal_interest_rate=0.05,
            day_count_convention=DayCountConvention.A360,
            interest_payment_cycle="1Y",
        )

        rf_obs = ConstantRiskFactorObserver(constant_value=0.05)

        contract = PrincipalAtMaturityContract(
            attributes=attrs,
            risk_factor_observer=rf_obs,
        )

        schedule = contract.generate_event_schedule()

        # Should have: 1 IED + IP at IED (payoff=0) + 2 IP (2025, 2026) + 1 IP at MD (2027) + 1 MD = 6 events
        # IP schedule from IED: 2024-01-15, 2025-01-15, 2026-01-15, 2027-01-15
        assert len(schedule.events) == 6

        # First event should be IED
        assert schedule.events[0].event_type == EventType.IED
        assert schedule.events[0].event_time == ActusDateTime(2024, 1, 15, 0, 0, 0)

        # Last event should be MD
        assert schedule.events[-1].event_type == EventType.MD
        assert schedule.events[-1].event_time == ActusDateTime(2027, 1, 15, 0, 0, 0)

        # There should be IP events (including IP at IED with payoff=0)
        ip_events = [e for e in schedule.events if e.event_type == EventType.IP]
        assert len(ip_events) == 4  # IED, 2025, 2026, 2027 (at MD)

    def test_generate_event_schedule_no_interest_cycle(self):
        """Test PAM event schedule without interest payment cycle."""
        attrs = ContractAttributes(
            contract_id="PAM-001",
            contract_type=ContractType.PAM,
            contract_role=ContractRole.RPA,
            status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
            initial_exchange_date=ActusDateTime(2024, 1, 15, 0, 0, 0),
            maturity_date=ActusDateTime(2029, 1, 15, 0, 0, 0),
            currency="USD",
            notional_principal=100000.0,
            nominal_interest_rate=0.05,
            day_count_convention=DayCountConvention.A360,
            interest_payment_cycle=None,  # No regular payments
        )

        rf_obs = ConstantRiskFactorObserver(constant_value=0.05)

        contract = PrincipalAtMaturityContract(
            attributes=attrs,
            risk_factor_observer=rf_obs,
        )

        schedule = contract.generate_event_schedule()

        # Should have only: 1 IED + 1 MD = 2 events
        assert len(schedule.events) == 2

        # First should be IED
        assert schedule.events[0].event_type == EventType.IED

        # Last should be MD
        assert schedule.events[1].event_type == EventType.MD

    def test_simulate_basic_loan(self):
        """Test PAM simulation with basic 5-year loan."""
        attrs = ContractAttributes(
            contract_id="PAM-001",
            contract_type=ContractType.PAM,
            contract_role=ContractRole.RPA,
            status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
            initial_exchange_date=ActusDateTime(2024, 1, 15, 0, 0, 0),
            maturity_date=ActusDateTime(2029, 1, 15, 0, 0, 0),
            currency="USD",
            notional_principal=100000.0,
            nominal_interest_rate=0.05,
            day_count_convention=DayCountConvention.A360,
            interest_payment_cycle="1Y",
        )

        rf_obs = ConstantRiskFactorObserver(constant_value=0.05)

        contract = PrincipalAtMaturityContract(
            attributes=attrs,
            risk_factor_observer=rf_obs,
        )

        result = contract.simulate()

        # Should have events
        assert len(result.events) > 0

        # First event should be IED with negative cashflow (disbursement)
        ied_event = result.events[0]
        assert ied_event.event_type == EventType.IED
        assert float(ied_event.payoff) < 0  # Negative = outflow

        # Last event should be MD with positive cashflow (repayment)
        md_event = result.events[-1]
        assert md_event.event_type == EventType.MD
        assert float(md_event.payoff) > 0  # Positive = inflow

        # Should have IP events in between (including IP at IED with payoff=0)
        ip_events = [e for e in result.events if e.event_type == EventType.IP]
        assert len(ip_events) == 6  # IP at IED + 5 annual payments

        # IP events after IED should have positive payoff (interest received by lender)
        for ip_event in ip_events:
            if ip_event.event_time != attrs.initial_exchange_date:
                assert float(ip_event.payoff) > 0

    def test_get_payoff_function(self):
        """Test get_payoff_function returns PAMPayoffFunction."""
        attrs = ContractAttributes(
            contract_id="PAM-001",
            contract_type=ContractType.PAM,
            contract_role=ContractRole.RPA,
            status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
            initial_exchange_date=ActusDateTime(2024, 1, 15, 0, 0, 0),
            maturity_date=ActusDateTime(2029, 1, 15, 0, 0, 0),
            currency="USD",
            notional_principal=100000.0,
        )

        rf_obs = ConstantRiskFactorObserver(constant_value=0.05)

        contract = PrincipalAtMaturityContract(
            attributes=attrs,
            risk_factor_observer=rf_obs,
        )

        pof = contract.get_payoff_function(EventType.IP)
        assert isinstance(pof, PAMPayoffFunction)

    def test_get_state_transition_function(self):
        """Test get_state_transition_function returns PAMStateTransitionFunction."""
        attrs = ContractAttributes(
            contract_id="PAM-001",
            contract_type=ContractType.PAM,
            contract_role=ContractRole.RPA,
            status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
            initial_exchange_date=ActusDateTime(2024, 1, 15, 0, 0, 0),
            maturity_date=ActusDateTime(2029, 1, 15, 0, 0, 0),
            currency="USD",
            notional_principal=100000.0,
        )

        rf_obs = ConstantRiskFactorObserver(constant_value=0.05)

        contract = PrincipalAtMaturityContract(
            attributes=attrs,
            risk_factor_observer=rf_obs,
        )

        stf = contract.get_state_transition_function(EventType.IP)
        assert isinstance(stf, PAMStateTransitionFunction)
