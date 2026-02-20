"""Unit tests for Linear Amortizer (LAM) contract implementation.

Tests the LinearAmortizerContract, LAMPayoffFunction, and LAMStateTransitionFunction
classes for correctness according to the ACTUS specification (Section 7.2).
"""

import jax.numpy as jnp
import pytest

from jactus.contracts.lam import (
    LAMPayoffFunction,
    LAMStateTransitionFunction,
    LinearAmortizerContract,
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
# Test LAMPayoffFunction
# ============================================================================


class TestLAMPayoffFunction:
    """Test LAMPayoffFunction class."""

    def test_initialization(self):
        """Test LAMPayoffFunction can be created."""
        pof = LAMPayoffFunction(
            contract_role=ContractRole.RPA,
            currency="USD",
            settlement_currency=None,
        )

        assert pof.contract_role == ContractRole.RPA
        assert pof.currency == "USD"

    def test_pof_ad_returns_zero(self):
        """Test POF_AD returns zero (no cashflow at analysis)."""
        pof = LAMPayoffFunction(
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
            prnxt=jnp.array(5000.0),
            ipcb=jnp.array(100000.0),
        )

        attrs = ContractAttributes(
            contract_id="LAM-001",
            contract_type=ContractType.LAM,
            contract_role=ContractRole.RPA,
            status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
            initial_exchange_date=ActusDateTime(2024, 1, 15, 0, 0, 0),
            maturity_date=ActusDateTime(2029, 1, 15, 0, 0, 0),
            currency="USD",
            notional_principal=100000.0,
            nominal_interest_rate=0.05,
            day_count_convention=DayCountConvention.A360,
            principal_redemption_cycle="1Y",
            next_principal_redemption_amount=5000.0,
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
        """Test POF_IED disburses principal (negative cashflow for borrower)."""
        pof = LAMPayoffFunction(
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
            contract_id="LAM-001",
            contract_type=ContractType.LAM,
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

    def test_pof_pr_pays_principal_redemption(self):
        """Test POF_PR pays principal redemption using prnxt state."""
        pof = LAMPayoffFunction(
            contract_role=ContractRole.RPA,
            currency="USD",
            settlement_currency=None,
        )

        state = ContractState(
            sd=ActusDateTime(2025, 1, 15, 0, 0, 0),
            tmd=ActusDateTime(2029, 1, 15, 0, 0, 0),
            nt=jnp.array(100000.0),
            ipnr=jnp.array(0.05),
            ipac=jnp.array(0.0),
            feac=jnp.array(0.0),
            nsc=jnp.array(1.0),
            isc=jnp.array(1.0),
            prnxt=jnp.array(5000.0),  # $5k principal redemption
            ipcb=jnp.array(100000.0),
        )

        attrs = ContractAttributes(
            contract_id="LAM-001",
            contract_type=ContractType.LAM,
            contract_role=ContractRole.RPA,
            status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
            initial_exchange_date=ActusDateTime(2024, 1, 15, 0, 0, 0),
            maturity_date=ActusDateTime(2029, 1, 15, 0, 0, 0),
            currency="USD",
            notional_principal=100000.0,
            nominal_interest_rate=0.05,
            day_count_convention=DayCountConvention.A360,
            principal_redemption_cycle="1Y",
            next_principal_redemption_amount=5000.0,
        )

        rf_obs = ConstantRiskFactorObserver(constant_value=0.05)

        payoff = pof.calculate_payoff(
            EventType.PR,
            state,
            attrs,
            ActusDateTime(2025, 1, 15, 0, 0, 0),
            rf_obs,
        )

        # Borrower pays $5k principal = positive cashflow from borrower perspective
        # POF_PR = R(CNTRL) × Nsc × Prnxt = 1 × 1.0 × 5000.0 = 5000.0
        assert float(payoff) == pytest.approx(5000.0)

    def test_pof_ip_calculates_interest_on_ipcb(self):
        """Test POF_IP calculates interest on IPCB, not current notional.

        This is the key difference between LAM and PAM:
        - PAM: Interest on current notional (Nt)
        - LAM: Interest on interest calculation base (Ipcb)
        """
        pof = LAMPayoffFunction(
            contract_role=ContractRole.RPA,
            currency="USD",
            settlement_currency=None,
        )

        # State after one year: notional reduced to 95k, but IPCB still 100k
        state = ContractState(
            sd=ActusDateTime(2024, 1, 15, 0, 0, 0),
            tmd=ActusDateTime(2029, 1, 15, 0, 0, 0),
            nt=jnp.array(95000.0),  # Current notional after PR
            ipnr=jnp.array(0.05),
            ipac=jnp.array(0.0),
            feac=jnp.array(0.0),
            nsc=jnp.array(1.0),
            isc=jnp.array(1.0),
            prnxt=jnp.array(5000.0),
            ipcb=jnp.array(100000.0),  # IPCB = initial notional
        )

        attrs = ContractAttributes(
            contract_id="LAM-001",
            contract_type=ContractType.LAM,
            contract_role=ContractRole.RPA,
            status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
            initial_exchange_date=ActusDateTime(2024, 1, 15, 0, 0, 0),
            maturity_date=ActusDateTime(2029, 1, 15, 0, 0, 0),
            currency="USD",
            notional_principal=100000.0,
            nominal_interest_rate=0.05,
            day_count_convention=DayCountConvention.A360,
            interest_payment_cycle="1Y",
            interest_calculation_base="NTIED",  # Fixed at IED
        )

        rf_obs = ConstantRiskFactorObserver(constant_value=0.05)

        payoff = pof.calculate_payoff(
            EventType.IP,
            state,
            attrs,
            ActusDateTime(2025, 1, 15, 0, 0, 0),
            rf_obs,
        )

        # Interest on IPCB (100k), NOT current notional (95k)
        # Year fraction A360 = actual days / 360 = 366/360 (2024 is leap year)
        # Interest = (366/360) × 0.05 × 100000 = 5083.33
        # POF_IP = R(CNTRL) × Isc × (Ipac + yf × Ipnr × Ipcb)
        #        = 1 × 1.0 × (0 + (366/360) × 0.05 × 100000)
        #        = 5083.33
        assert float(payoff) == pytest.approx(5083.33, rel=0.01)

    def test_pof_md_returns_remaining_principal(self):
        """Test POF_MD returns remaining principal at maturity."""
        pof = LAMPayoffFunction(
            contract_role=ContractRole.RPA,
            currency="USD",
            settlement_currency=None,
        )

        # At maturity with small remaining balance
        state = ContractState(
            sd=ActusDateTime(2028, 1, 15, 0, 0, 0),
            tmd=ActusDateTime(2029, 1, 15, 0, 0, 0),
            nt=jnp.array(5000.0),  # Last principal payment
            ipnr=jnp.array(0.05),
            ipac=jnp.array(250.0),  # Some accrued interest
            feac=jnp.array(0.0),
            nsc=jnp.array(1.0),
            isc=jnp.array(1.0),
            prnxt=jnp.array(5000.0),
            ipcb=jnp.array(5000.0),
        )

        attrs = ContractAttributes(
            contract_id="LAM-001",
            contract_type=ContractType.LAM,
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

        # POF_MD = R(CNTRL) × (Nsc × Nt + Isc × Ipac + Feac)
        #        = 1 × (1.0 × 5000 + 1.0 × 250 + 0)
        #        = 5250.0
        assert float(payoff) == pytest.approx(5250.0)


# ============================================================================
# Test LAMStateTransitionFunction
# ============================================================================


class TestLAMStateTransitionFunction:
    """Test LAMStateTransitionFunction class."""

    def test_initialization(self):
        """Test LAMStateTransitionFunction can be created."""
        stf = LAMStateTransitionFunction()
        assert stf is not None

    def test_stf_ied_initializes_state_with_ipcb_ntied(self):
        """Test STF_IED initializes state with IPCB=NTIED mode."""
        stf = LAMStateTransitionFunction()

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
            contract_id="LAM-001",
            contract_type=ContractType.LAM,
            contract_role=ContractRole.RPA,
            status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
            initial_exchange_date=ActusDateTime(2024, 1, 15, 0, 0, 0),
            maturity_date=ActusDateTime(2029, 1, 15, 0, 0, 0),
            currency="USD",
            notional_principal=100000.0,
            nominal_interest_rate=0.05,
            day_count_convention=DayCountConvention.A360,
            interest_calculation_base="NTIED",  # Fixed at IED
            next_principal_redemption_amount=5000.0,
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
        assert float(new_state.ipac) == pytest.approx(0.0)
        # IPCB should be set to initial notional for NTIED mode
        assert new_state.ipcb is not None
        assert float(new_state.ipcb) == pytest.approx(100000.0)
        # Prnxt should be set
        assert new_state.prnxt is not None
        assert float(new_state.prnxt) == pytest.approx(5000.0)

    def test_stf_ied_initializes_state_with_ipcb_nt(self):
        """Test STF_IED initializes state with IPCB=NT mode (lagging)."""
        stf = LAMStateTransitionFunction()

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
            contract_id="LAM-001",
            contract_type=ContractType.LAM,
            contract_role=ContractRole.RPA,
            status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
            initial_exchange_date=ActusDateTime(2024, 1, 15, 0, 0, 0),
            maturity_date=ActusDateTime(2029, 1, 15, 0, 0, 0),
            currency="USD",
            notional_principal=100000.0,
            nominal_interest_rate=0.05,
            day_count_convention=DayCountConvention.A360,
            interest_calculation_base="NT",  # Lagging notional
            next_principal_redemption_amount=5000.0,
        )

        rf_obs = ConstantRiskFactorObserver(constant_value=0.05)

        new_state = stf.transition_state(
            EventType.IED,
            initial_state,
            attrs,
            ActusDateTime(2024, 1, 15, 0, 0, 0),
            rf_obs,
        )

        assert float(new_state.nt) == pytest.approx(100000.0)
        # IPCB = NT initially, will lag after PR events
        assert float(new_state.ipcb) == pytest.approx(100000.0)

    def test_stf_ied_initializes_state_with_ipcb_ntl(self):
        """Test STF_IED initializes state with IPCB=NTL mode (with IPCB events)."""
        stf = LAMStateTransitionFunction()

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
            contract_id="LAM-001",
            contract_type=ContractType.LAM,
            contract_role=ContractRole.RPA,
            status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
            initial_exchange_date=ActusDateTime(2024, 1, 15, 0, 0, 0),
            maturity_date=ActusDateTime(2029, 1, 15, 0, 0, 0),
            currency="USD",
            notional_principal=100000.0,
            nominal_interest_rate=0.05,
            day_count_convention=DayCountConvention.A360,
            interest_calculation_base="NTL",  # With IPCB events
            next_principal_redemption_amount=5000.0,
        )

        rf_obs = ConstantRiskFactorObserver(constant_value=0.05)

        new_state = stf.transition_state(
            EventType.IED,
            initial_state,
            attrs,
            ActusDateTime(2024, 1, 15, 0, 0, 0),
            rf_obs,
        )

        assert float(new_state.nt) == pytest.approx(100000.0)
        # IPCB initialized, will be updated at IPCB events
        assert float(new_state.ipcb) == pytest.approx(100000.0)

    def test_stf_pr_reduces_notional_and_updates_ipcb_nt_mode(self):
        """Test STF_PR reduces notional and updates IPCB in NT mode."""
        stf = LAMStateTransitionFunction()

        state = ContractState(
            sd=ActusDateTime(2024, 1, 15, 0, 0, 0),
            tmd=ActusDateTime(2029, 1, 15, 0, 0, 0),
            nt=jnp.array(100000.0),
            ipnr=jnp.array(0.05),
            ipac=jnp.array(0.0),
            feac=jnp.array(0.0),
            nsc=jnp.array(1.0),
            isc=jnp.array(1.0),
            prnxt=jnp.array(5000.0),
            ipcb=jnp.array(100000.0),
        )

        attrs = ContractAttributes(
            contract_id="LAM-001",
            contract_type=ContractType.LAM,
            contract_role=ContractRole.RPA,
            status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
            initial_exchange_date=ActusDateTime(2024, 1, 15, 0, 0, 0),
            maturity_date=ActusDateTime(2029, 1, 15, 0, 0, 0),
            currency="USD",
            notional_principal=100000.0,
            nominal_interest_rate=0.05,
            day_count_convention=DayCountConvention.A360,
            interest_calculation_base="NT",  # NT mode: IPCB follows NT
            next_principal_redemption_amount=5000.0,
        )

        rf_obs = ConstantRiskFactorObserver(constant_value=0.05)

        new_state = stf.transition_state(
            EventType.PR,
            state,
            attrs,
            ActusDateTime(2025, 1, 15, 0, 0, 0),
            rf_obs,
        )

        assert new_state.sd == ActusDateTime(2025, 1, 15, 0, 0, 0)
        # Notional reduced by prnxt
        assert float(new_state.nt) == pytest.approx(95000.0)
        # Interest accrued: (366/360) year × 0.05 × 100000 = 5083.33
        assert float(new_state.ipac) == pytest.approx(5083.33, rel=0.01)
        # In NT mode, IPCB follows new notional
        assert float(new_state.ipcb) == pytest.approx(95000.0)

    def test_stf_pr_reduces_notional_keeps_ipcb_ntied_mode(self):
        """Test STF_PR reduces notional but keeps IPCB fixed in NTIED mode."""
        stf = LAMStateTransitionFunction()

        state = ContractState(
            sd=ActusDateTime(2024, 1, 15, 0, 0, 0),
            tmd=ActusDateTime(2029, 1, 15, 0, 0, 0),
            nt=jnp.array(100000.0),
            ipnr=jnp.array(0.05),
            ipac=jnp.array(0.0),
            feac=jnp.array(0.0),
            nsc=jnp.array(1.0),
            isc=jnp.array(1.0),
            prnxt=jnp.array(5000.0),
            ipcb=jnp.array(100000.0),  # Fixed at IED
        )

        attrs = ContractAttributes(
            contract_id="LAM-001",
            contract_type=ContractType.LAM,
            contract_role=ContractRole.RPA,
            status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
            initial_exchange_date=ActusDateTime(2024, 1, 15, 0, 0, 0),
            maturity_date=ActusDateTime(2029, 1, 15, 0, 0, 0),
            currency="USD",
            notional_principal=100000.0,
            nominal_interest_rate=0.05,
            day_count_convention=DayCountConvention.A360,
            interest_calculation_base="NTIED",  # NTIED: IPCB stays fixed
            next_principal_redemption_amount=5000.0,
        )

        rf_obs = ConstantRiskFactorObserver(constant_value=0.05)

        new_state = stf.transition_state(
            EventType.PR,
            state,
            attrs,
            ActusDateTime(2025, 1, 15, 0, 0, 0),
            rf_obs,
        )

        # Notional reduced
        assert float(new_state.nt) == pytest.approx(95000.0)
        # IPCB tracks current notional (NTIED behaves like NT per ACTUS tests)
        assert float(new_state.ipcb) == pytest.approx(95000.0)

    def test_stf_ipcb_resets_ipcb_to_current_notional(self):
        """Test STF_IPCB resets IPCB to current notional (NTL mode)."""
        stf = LAMStateTransitionFunction()

        state = ContractState(
            sd=ActusDateTime(2025, 1, 15, 0, 0, 0),
            tmd=ActusDateTime(2029, 1, 15, 0, 0, 0),
            nt=jnp.array(95000.0),  # After one PR event
            ipnr=jnp.array(0.05),
            ipac=jnp.array(0.0),
            feac=jnp.array(0.0),
            nsc=jnp.array(1.0),
            isc=jnp.array(1.0),
            prnxt=jnp.array(5000.0),
            ipcb=jnp.array(100000.0),  # Old IPCB
        )

        attrs = ContractAttributes(
            contract_id="LAM-001",
            contract_type=ContractType.LAM,
            contract_role=ContractRole.RPA,
            status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
            initial_exchange_date=ActusDateTime(2024, 1, 15, 0, 0, 0),
            maturity_date=ActusDateTime(2029, 1, 15, 0, 0, 0),
            currency="USD",
            notional_principal=100000.0,
            nominal_interest_rate=0.05,
            day_count_convention=DayCountConvention.A360,
            interest_calculation_base="NTL",
            next_principal_redemption_amount=5000.0,
        )

        rf_obs = ConstantRiskFactorObserver(constant_value=0.05)

        new_state = stf.transition_state(
            EventType.IPCB,
            state,
            attrs,
            ActusDateTime(2025, 7, 15, 0, 0, 0),
            rf_obs,
        )

        assert new_state.sd == ActusDateTime(2025, 7, 15, 0, 0, 0)
        # Notional unchanged
        assert float(new_state.nt) == pytest.approx(95000.0)
        # IPCB reset to current notional
        assert float(new_state.ipcb) == pytest.approx(95000.0)
        # Interest accrued: (181/360) × 0.05 × 100000 = 2513.89
        assert float(new_state.ipac) == pytest.approx(2513.89, rel=0.01)

    def test_stf_ip_resets_ipac_to_zero(self):
        """Test STF_IP resets accrued interest to zero after payment."""
        stf = LAMStateTransitionFunction()

        state = ContractState(
            sd=ActusDateTime(2024, 1, 15, 0, 0, 0),
            tmd=ActusDateTime(2029, 1, 15, 0, 0, 0),
            nt=jnp.array(100000.0),
            ipnr=jnp.array(0.05),
            ipac=jnp.array(0.0),
            feac=jnp.array(0.0),
            nsc=jnp.array(1.0),
            isc=jnp.array(1.0),
            prnxt=jnp.array(5000.0),
            ipcb=jnp.array(100000.0),
        )

        attrs = ContractAttributes(
            contract_id="LAM-001",
            contract_type=ContractType.LAM,
            contract_role=ContractRole.RPA,
            status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
            initial_exchange_date=ActusDateTime(2024, 1, 15, 0, 0, 0),
            maturity_date=ActusDateTime(2029, 1, 15, 0, 0, 0),
            currency="USD",
            notional_principal=100000.0,
            nominal_interest_rate=0.05,
            day_count_convention=DayCountConvention.A360,
            interest_payment_cycle="1Y",
            next_principal_redemption_amount=5000.0,
        )

        rf_obs = ConstantRiskFactorObserver(constant_value=0.05)

        new_state = stf.transition_state(
            EventType.IP,
            state,
            attrs,
            ActusDateTime(2025, 1, 15, 0, 0, 0),
            rf_obs,
        )

        assert new_state.sd == ActusDateTime(2025, 1, 15, 0, 0, 0)
        # IPAC reset to zero after payment
        assert float(new_state.ipac) == pytest.approx(0.0)
        # Notional and IPCB unchanged
        assert float(new_state.nt) == pytest.approx(100000.0)
        assert float(new_state.ipcb) == pytest.approx(100000.0)


# ============================================================================
# Test LinearAmortizerContract
# ============================================================================


class TestLinearAmortizerContract:
    """Test LinearAmortizerContract class."""

    def test_initialization(self):
        """Test LinearAmortizerContract can be created."""
        attrs = ContractAttributes(
            contract_id="LAM-001",
            contract_type=ContractType.LAM,
            contract_role=ContractRole.RPA,
            status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
            initial_exchange_date=ActusDateTime(2024, 1, 15, 0, 0, 0),
            maturity_date=ActusDateTime(2029, 1, 15, 0, 0, 0),
            currency="USD",
            notional_principal=100000.0,
            nominal_interest_rate=0.05,
            day_count_convention=DayCountConvention.A360,
            principal_redemption_cycle="1Y",
            next_principal_redemption_amount=5000.0,
        )

        rf_obs = ConstantRiskFactorObserver(constant_value=0.05)
        contract = LinearAmortizerContract(attributes=attrs, risk_factor_observer=rf_obs)

        assert contract.attributes.contract_id == "LAM-001"
        assert contract.attributes.contract_type == ContractType.LAM

    def test_pr_schedule_generation(self):
        """Test that PR events are generated according to principal_redemption_cycle."""
        attrs = ContractAttributes(
            contract_id="LAM-001",
            contract_type=ContractType.LAM,
            contract_role=ContractRole.RPA,
            status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
            initial_exchange_date=ActusDateTime(2024, 1, 15, 0, 0, 0),
            maturity_date=ActusDateTime(2029, 1, 15, 0, 0, 0),
            currency="USD",
            notional_principal=100000.0,
            nominal_interest_rate=0.05,
            day_count_convention=DayCountConvention.A360,
            principal_redemption_cycle="1Y",  # Annual PR events
            next_principal_redemption_amount=5000.0,
        )

        rf_obs = ConstantRiskFactorObserver(constant_value=0.05)
        contract = LinearAmortizerContract(attributes=attrs, risk_factor_observer=rf_obs)

        schedule = contract.generate_event_schedule()

        # Find PR events
        pr_events = [e for e in schedule.events if e.event_type == EventType.PR]

        # PR at PRANX (=IED), then 2025-2028 = 5 PR events total (2029 = MD)
        assert len(pr_events) == 5
        assert pr_events[0].event_time.year == 2024  # PR at IED/PRANX
        assert pr_events[1].event_time.year == 2025
        assert pr_events[2].event_time.year == 2026
        assert pr_events[3].event_time.year == 2027
        assert pr_events[4].event_time.year == 2028

    def test_ipcb_schedule_generation_ntl_mode(self):
        """Test that IPCB events are generated only when IPCB='NTL'."""
        attrs = ContractAttributes(
            contract_id="LAM-001",
            contract_type=ContractType.LAM,
            contract_role=ContractRole.RPA,
            status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
            initial_exchange_date=ActusDateTime(2024, 1, 15, 0, 0, 0),
            maturity_date=ActusDateTime(2029, 1, 15, 0, 0, 0),
            currency="USD",
            notional_principal=100000.0,
            nominal_interest_rate=0.05,
            day_count_convention=DayCountConvention.A360,
            interest_calculation_base="NTL",
            interest_calculation_base_cycle="6M",  # Semi-annual IPCB events
        )

        rf_obs = ConstantRiskFactorObserver(constant_value=0.05)
        contract = LinearAmortizerContract(attributes=attrs, risk_factor_observer=rf_obs)

        schedule = contract.generate_event_schedule()

        # Find IPCB events
        ipcb_events = [e for e in schedule.events if e.event_type == EventType.IPCB]

        # Should have IPCB events (2x per year for 5 years = 10, but stops before MD)
        assert len(ipcb_events) > 0

    def test_no_ipcb_schedule_when_not_ntl(self):
        """Test that no IPCB events generated when IPCB != 'NTL'."""
        attrs = ContractAttributes(
            contract_id="LAM-001",
            contract_type=ContractType.LAM,
            contract_role=ContractRole.RPA,
            status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
            initial_exchange_date=ActusDateTime(2024, 1, 15, 0, 0, 0),
            maturity_date=ActusDateTime(2029, 1, 15, 0, 0, 0),
            currency="USD",
            notional_principal=100000.0,
            nominal_interest_rate=0.05,
            day_count_convention=DayCountConvention.A360,
            interest_calculation_base="NTIED",  # Not NTL
        )

        rf_obs = ConstantRiskFactorObserver(constant_value=0.05)
        contract = LinearAmortizerContract(attributes=attrs, risk_factor_observer=rf_obs)

        schedule = contract.generate_event_schedule()

        # Find IPCB events
        ipcb_events = [e for e in schedule.events if e.event_type == EventType.IPCB]

        # Should have NO IPCB events
        assert len(ipcb_events) == 0

    def test_simulate_basic_lam_loan(self):
        """Test complete LAM simulation with annual PR and IP events."""
        attrs = ContractAttributes(
            contract_id="LAM-001",
            contract_type=ContractType.LAM,
            contract_role=ContractRole.RPA,
            status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
            initial_exchange_date=ActusDateTime(2024, 1, 15, 0, 0, 0),
            maturity_date=ActusDateTime(2029, 1, 15, 0, 0, 0),
            currency="USD",
            notional_principal=100000.0,
            nominal_interest_rate=0.05,
            day_count_convention=DayCountConvention.A360,
            principal_redemption_cycle="1Y",  # Annual principal payments
            interest_payment_cycle="1Y",  # Annual interest payments
            next_principal_redemption_amount=5000.0,  # $5k per year
            interest_calculation_base="NTIED",  # Fixed at IED
        )

        rf_obs = ConstantRiskFactorObserver(constant_value=0.05)
        contract = LinearAmortizerContract(attributes=attrs, risk_factor_observer=rf_obs)

        result = contract.simulate()

        # Should have events: IED + 4×(PR+IP) + MD = 1 + 8 + 1 = 10 events
        # Actually: IED, then 4 years of PR/IP, then MD
        assert len(result.events) >= 9

        # Check IED event
        ied_event = result.events[0]
        assert ied_event.event_type == EventType.IED
        assert float(ied_event.payoff) == pytest.approx(-100000.0)  # Disbursement

        # Check that notional decreases over time
        pr_events = [e for e in result.events if e.event_type == EventType.PR]
        for pr_event in pr_events:
            assert float(pr_event.payoff) == pytest.approx(5000.0)  # $5k principal payment

        # Check that interest payments exist and decrease (NTIED tracks NT per ACTUS spec)
        ip_events = [
            e for e in result.events
            if e.event_type == EventType.IP and e.event_time != attrs.initial_exchange_date
        ]
        assert len(ip_events) >= 3
        # First IP: PR fires at IED reducing NT to 95000, then first year interest
        # 95000 * 0.05 * 366/360 ≈ 4829.17 (2024 is a leap year)
        assert float(ip_events[0].payoff) == pytest.approx(4829.17, rel=0.01)

    def test_notional_reaches_zero_at_maturity(self):
        """Test that notional is fully amortized by maturity."""
        # 20k principal, 5k per year, 4 years = fully amortized
        attrs = ContractAttributes(
            contract_id="LAM-001",
            contract_type=ContractType.LAM,
            contract_role=ContractRole.RPA,
            status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
            initial_exchange_date=ActusDateTime(2024, 1, 15, 0, 0, 0),
            maturity_date=ActusDateTime(2028, 1, 15, 0, 0, 0),  # 4 years
            currency="USD",
            notional_principal=20000.0,
            nominal_interest_rate=0.05,
            day_count_convention=DayCountConvention.A360,
            principal_redemption_cycle="1Y",
            next_principal_redemption_amount=5000.0,  # $5k per year × 4 = $20k
        )

        rf_obs = ConstantRiskFactorObserver(constant_value=0.05)
        contract = LinearAmortizerContract(attributes=attrs, risk_factor_observer=rf_obs)

        result = contract.simulate()

        # Find MD event
        md_event = [e for e in result.events if e.event_type == EventType.MD][0]

        # Check number of PR events
        pr_events = [e for e in result.events if e.event_type == EventType.PR]
        # PRANX defaults to IED, so PR at 2024, 2025, 2026, 2027 (4 PRs before MD=2028)
        assert len(pr_events) == 4

        # At MD, remaining notional should be 0 (20k - 4×5k)
        assert md_event.state_pre is not None
        assert float(md_event.state_pre.nt) == pytest.approx(0.0, abs=1e-3)


# ============================================================================
# Run Tests
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
