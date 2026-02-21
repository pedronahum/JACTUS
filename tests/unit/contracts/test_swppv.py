"""Unit tests for SWPPV (Plain Vanilla Interest Rate Swap) contract."""

import jax.numpy as jnp
import pytest

from jactus.contracts import PlainVanillaSwapContract, create_contract
from jactus.core import (
    ActusDateTime,
    ContractAttributes,
    ContractRole,
    ContractType,
    EventType,
)
from jactus.observers import ConstantRiskFactorObserver


class TestSWPPVInitialization:
    """Test SWPPV contract initialization and validation."""

    def test_swppv_contract_creation(self):
        """Test successful SWPPV contract creation."""
        attrs = ContractAttributes(
            contract_id="SWAP001",
            contract_type=ContractType.SWPPV,
            contract_role=ContractRole.RPA,
            status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
            contract_deal_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
            initial_exchange_date=ActusDateTime(2024, 1, 15, 0, 0, 0),
            maturity_date=ActusDateTime(2026, 1, 15, 0, 0, 0),
            notional_principal=1000000.0,
            nominal_interest_rate=0.05,  # Fixed: 5%
            nominal_interest_rate_2=0.03,  # Floating initial: 3%
            interest_payment_cycle="6M",
            rate_reset_cycle="3M",
            interest_calculation_base_anchor=ActusDateTime(2024, 1, 15, 0, 0, 0),
            rate_reset_anchor=ActusDateTime(2024, 1, 15, 0, 0, 0),
        )

        rf_obs = ConstantRiskFactorObserver(constant_value=0.04)
        swap = PlainVanillaSwapContract(attrs, rf_obs)

        assert swap is not None
        assert swap.attributes.contract_type == ContractType.SWPPV

    def test_swppv_requires_notional(self):
        """Test that SWPPV requires notional_principal."""
        attrs = ContractAttributes(
            contract_id="SWAP001",
            contract_type=ContractType.SWPPV,
            contract_role=ContractRole.RPA,
            status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
            contract_deal_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
            maturity_date=ActusDateTime(2026, 1, 15, 0, 0, 0),
            # Missing notional_principal
            nominal_interest_rate=0.05,
            nominal_interest_rate_2=0.03,
            interest_payment_cycle="6M",
            rate_reset_cycle="3M",
            interest_calculation_base_anchor=ActusDateTime(2024, 1, 15, 0, 0, 0),
            rate_reset_anchor=ActusDateTime(2024, 1, 15, 0, 0, 0),
        )

        rf_obs = ConstantRiskFactorObserver(constant_value=0.04)

        with pytest.raises(ValueError, match="notional_principal.*required"):
            PlainVanillaSwapContract(attrs, rf_obs)

    def test_swppv_requires_fixed_rate(self):
        """Test that SWPPV requires nominal_interest_rate (fixed leg)."""
        attrs = ContractAttributes(
            contract_id="SWAP001",
            contract_type=ContractType.SWPPV,
            contract_role=ContractRole.RPA,
            status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
            contract_deal_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
            maturity_date=ActusDateTime(2026, 1, 15, 0, 0, 0),
            notional_principal=1000000.0,
            # Missing nominal_interest_rate
            nominal_interest_rate_2=0.03,
            interest_payment_cycle="6M",
            rate_reset_cycle="3M",
            interest_calculation_base_anchor=ActusDateTime(2024, 1, 15, 0, 0, 0),
            rate_reset_anchor=ActusDateTime(2024, 1, 15, 0, 0, 0),
        )

        rf_obs = ConstantRiskFactorObserver(constant_value=0.04)

        with pytest.raises(ValueError, match="nominal_interest_rate.*required"):
            PlainVanillaSwapContract(attrs, rf_obs)

    def test_swppv_requires_floating_rate(self):
        """Test that SWPPV requires nominal_interest_rate_2 (floating leg)."""
        attrs = ContractAttributes(
            contract_id="SWAP001",
            contract_type=ContractType.SWPPV,
            contract_role=ContractRole.RPA,
            status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
            contract_deal_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
            maturity_date=ActusDateTime(2026, 1, 15, 0, 0, 0),
            notional_principal=1000000.0,
            nominal_interest_rate=0.05,
            # Missing nominal_interest_rate_2
            interest_payment_cycle="6M",
            rate_reset_cycle="3M",
            interest_calculation_base_anchor=ActusDateTime(2024, 1, 15, 0, 0, 0),
            rate_reset_anchor=ActusDateTime(2024, 1, 15, 0, 0, 0),
        )

        rf_obs = ConstantRiskFactorObserver(constant_value=0.04)

        with pytest.raises(ValueError, match="nominal_interest_rate_2.*required"):
            PlainVanillaSwapContract(attrs, rf_obs)

    def test_swppv_requires_payment_cycle(self):
        """Test that SWPPV requires interest_payment_cycle."""
        attrs = ContractAttributes(
            contract_id="SWAP001",
            contract_type=ContractType.SWPPV,
            contract_role=ContractRole.RPA,
            status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
            contract_deal_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
            maturity_date=ActusDateTime(2026, 1, 15, 0, 0, 0),
            notional_principal=1000000.0,
            nominal_interest_rate=0.05,
            nominal_interest_rate_2=0.03,
            # Missing interest_payment_cycle
            rate_reset_cycle="3M",
            interest_calculation_base_anchor=ActusDateTime(2024, 1, 15, 0, 0, 0),
            rate_reset_anchor=ActusDateTime(2024, 1, 15, 0, 0, 0),
        )

        rf_obs = ConstantRiskFactorObserver(constant_value=0.04)

        with pytest.raises(ValueError, match="interest_payment_cycle.*required"):
            PlainVanillaSwapContract(attrs, rf_obs)

    def test_swppv_rate_reset_cycle_optional(self):
        """Test that SWPPV does not require rate_reset_cycle (it is optional)."""
        attrs = ContractAttributes(
            contract_id="SWAP001",
            contract_type=ContractType.SWPPV,
            contract_role=ContractRole.RPA,
            status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
            contract_deal_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
            maturity_date=ActusDateTime(2026, 1, 15, 0, 0, 0),
            notional_principal=1000000.0,
            nominal_interest_rate=0.05,
            nominal_interest_rate_2=0.03,
            interest_payment_cycle="6M",
            # No rate_reset_cycle — should still create successfully
            interest_calculation_base_anchor=ActusDateTime(2024, 1, 15, 0, 0, 0),
            rate_reset_anchor=ActusDateTime(2024, 1, 15, 0, 0, 0),
        )

        rf_obs = ConstantRiskFactorObserver(constant_value=0.04)

        # Should create successfully without rate_reset_cycle
        swap = PlainVanillaSwapContract(attrs, rf_obs)
        assert swap is not None
        assert swap.attributes.contract_type == ContractType.SWPPV

    def test_swppv_wrong_contract_type(self):
        """Test that wrong contract type raises error."""
        attrs = ContractAttributes(
            contract_id="SWAP001",
            contract_type=ContractType.PAM,  # Wrong type!
            contract_role=ContractRole.RPA,
            status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
            contract_deal_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
            maturity_date=ActusDateTime(2026, 1, 15, 0, 0, 0),
            notional_principal=1000000.0,
            nominal_interest_rate=0.05,
            nominal_interest_rate_2=0.03,
            interest_payment_cycle="6M",
            rate_reset_cycle="3M",
            interest_calculation_base_anchor=ActusDateTime(2024, 1, 15, 0, 0, 0),
            rate_reset_anchor=ActusDateTime(2024, 1, 15, 0, 0, 0),
        )

        rf_obs = ConstantRiskFactorObserver(constant_value=0.04)

        with pytest.raises(ValueError, match="Expected contract_type=SWPPV"):
            PlainVanillaSwapContract(attrs, rf_obs)


class TestSWPPVEventSchedule:
    """Test SWPPV event schedule generation."""

    def test_swppv_generates_ied_event(self):
        """Test that SWPPV generates IED event."""
        attrs = ContractAttributes(
            contract_id="SWAP001",
            contract_type=ContractType.SWPPV,
            contract_role=ContractRole.RPA,
            status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
            contract_deal_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
            initial_exchange_date=ActusDateTime(2024, 1, 15, 0, 0, 0),
            maturity_date=ActusDateTime(2026, 1, 15, 0, 0, 0),
            notional_principal=1000000.0,
            nominal_interest_rate=0.05,
            nominal_interest_rate_2=0.03,
            interest_payment_cycle="6M",
            rate_reset_cycle="3M",
            interest_calculation_base_anchor=ActusDateTime(2024, 1, 15, 0, 0, 0),
            rate_reset_anchor=ActusDateTime(2024, 1, 15, 0, 0, 0),
        )

        rf_obs = ConstantRiskFactorObserver(constant_value=0.04)
        swap = PlainVanillaSwapContract(attrs, rf_obs)

        schedule = swap.generate_event_schedule()
        event_types = [e.event_type for e in schedule.events]

        assert EventType.IED in event_types

    def test_swppv_generates_ip_events(self):
        """Test that SWPPV generates interest payment events.

        With default delivery_settlement="D", SWPPV generates separate
        IPFX (fixed leg) and IPFL (floating leg) events instead of IP events.
        """
        attrs = ContractAttributes(
            contract_id="SWAP001",
            contract_type=ContractType.SWPPV,
            contract_role=ContractRole.RPA,
            status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
            contract_deal_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
            initial_exchange_date=ActusDateTime(2024, 1, 15, 0, 0, 0),
            maturity_date=ActusDateTime(2025, 1, 15, 0, 0, 0),
            notional_principal=1000000.0,
            nominal_interest_rate=0.05,
            nominal_interest_rate_2=0.03,
            interest_payment_cycle="6M",  # Semi-annual
            rate_reset_cycle="3M",
            interest_calculation_base_anchor=ActusDateTime(2024, 1, 15, 0, 0, 0),
            rate_reset_anchor=ActusDateTime(2024, 1, 15, 0, 0, 0),
        )

        rf_obs = ConstantRiskFactorObserver(constant_value=0.04)
        swap = PlainVanillaSwapContract(attrs, rf_obs)

        schedule = swap.generate_event_schedule()
        ipfx_events = [e for e in schedule.events if e.event_type == EventType.IPFX]
        ipfl_events = [e for e in schedule.events if e.event_type == EventType.IPFL]

        # Default DS="D" generates IPFX/IPFL pairs, not IP events
        # Should have 2 IPFX and 2 IPFL events for 1-year swap with semi-annual payments
        assert len(ipfx_events) >= 2
        assert len(ipfl_events) >= 2

    def test_swppv_generates_rr_events(self):
        """Test that SWPPV generates RR (Rate Reset) events."""
        attrs = ContractAttributes(
            contract_id="SWAP001",
            contract_type=ContractType.SWPPV,
            contract_role=ContractRole.RPA,
            status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
            contract_deal_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
            initial_exchange_date=ActusDateTime(2024, 1, 15, 0, 0, 0),
            maturity_date=ActusDateTime(2025, 1, 15, 0, 0, 0),
            notional_principal=1000000.0,
            nominal_interest_rate=0.05,
            nominal_interest_rate_2=0.03,
            interest_payment_cycle="6M",
            rate_reset_cycle="3M",  # Quarterly resets
            interest_calculation_base_anchor=ActusDateTime(2024, 1, 15, 0, 0, 0),
            rate_reset_anchor=ActusDateTime(2024, 1, 15, 0, 0, 0),
        )

        rf_obs = ConstantRiskFactorObserver(constant_value=0.04)
        swap = PlainVanillaSwapContract(attrs, rf_obs)

        schedule = swap.generate_event_schedule()
        rr_events = [e for e in schedule.events if e.event_type == EventType.RR]

        # Should have 4 RR events for 1-year swap with quarterly resets
        assert len(rr_events) >= 4

    def test_swppv_generates_md_event(self):
        """Test that SWPPV generates MD (Maturity Date) event."""
        attrs = ContractAttributes(
            contract_id="SWAP001",
            contract_type=ContractType.SWPPV,
            contract_role=ContractRole.RPA,
            status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
            maturity_date=ActusDateTime(2026, 1, 15, 0, 0, 0),
            notional_principal=1000000.0,
            nominal_interest_rate=0.05,
            nominal_interest_rate_2=0.03,
            interest_payment_cycle="6M",
            rate_reset_cycle="3M",
        )

        rf_obs = ConstantRiskFactorObserver(constant_value=0.04)
        swap = PlainVanillaSwapContract(attrs, rf_obs)

        schedule = swap.generate_event_schedule()

        md_events = [e for e in schedule.events if e.event_type == EventType.MD]
        assert len(md_events) == 1
        assert md_events[0].event_time == attrs.maturity_date

    def test_swppv_generates_ad_events(self):
        """Test that SWPPV generates AD (Analysis Date) events."""
        attrs = ContractAttributes(
            contract_id="SWAP001",
            contract_type=ContractType.SWPPV,
            contract_role=ContractRole.RPA,
            status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
            maturity_date=ActusDateTime(2026, 1, 15, 0, 0, 0),
            notional_principal=1000000.0,
            nominal_interest_rate=0.05,
            nominal_interest_rate_2=0.03,
            interest_payment_cycle="6M",
            rate_reset_cycle="3M",
            analysis_dates=[
                ActusDateTime(2024, 3, 1, 0, 0, 0),
                ActusDateTime(2024, 6, 1, 0, 0, 0),
            ],
        )

        rf_obs = ConstantRiskFactorObserver(constant_value=0.04)
        swap = PlainVanillaSwapContract(attrs, rf_obs)

        schedule = swap.generate_event_schedule()

        ad_events = [e for e in schedule.events if e.event_type == EventType.AD]
        assert len(ad_events) == 2

    def test_swppv_generates_td_events(self):
        """Test that SWPPV generates TD (Termination Date) events."""
        attrs = ContractAttributes(
            contract_id="SWAP001",
            contract_type=ContractType.SWPPV,
            contract_role=ContractRole.RPA,
            status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
            maturity_date=ActusDateTime(2026, 1, 15, 0, 0, 0),
            notional_principal=1000000.0,
            nominal_interest_rate=0.05,
            nominal_interest_rate_2=0.03,
            interest_payment_cycle="6M",
            rate_reset_cycle="3M",
            termination_date=ActusDateTime(2025, 6, 1, 0, 0, 0),
        )

        rf_obs = ConstantRiskFactorObserver(constant_value=0.04)
        swap = PlainVanillaSwapContract(attrs, rf_obs)

        schedule = swap.generate_event_schedule()

        td_events = [e for e in schedule.events if e.event_type == EventType.TD]
        assert len(td_events) == 1
        assert td_events[0].event_time == attrs.termination_date


class TestSWPPVStateInitialization:
    """Test SWPPV state initialization."""

    def test_swppv_initialize_state(self):
        """Test SWPPV state initialization."""
        attrs = ContractAttributes(
            contract_id="SWAP001",
            contract_type=ContractType.SWPPV,
            contract_role=ContractRole.RPA,
            status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
            contract_deal_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
            initial_exchange_date=ActusDateTime(2024, 1, 15, 0, 0, 0),
            maturity_date=ActusDateTime(2026, 1, 15, 0, 0, 0),
            notional_principal=1000000.0,
            nominal_interest_rate=0.05,
            nominal_interest_rate_2=0.03,
            interest_payment_cycle="6M",
            rate_reset_cycle="3M",
            interest_calculation_base_anchor=ActusDateTime(2024, 1, 15, 0, 0, 0),
            rate_reset_anchor=ActusDateTime(2024, 1, 15, 0, 0, 0),
        )

        rf_obs = ConstantRiskFactorObserver(constant_value=0.04)
        swap = PlainVanillaSwapContract(attrs, rf_obs)

        state = swap.initialize_state()

        # Check state initialized with floating rate (IPNR2)
        assert float(state.ipnr) == pytest.approx(0.03, abs=0.0001)
        assert float(state.ipac) == pytest.approx(0.0, abs=0.0001)
        assert float(state.ipac1) == pytest.approx(0.0, abs=0.0001)
        assert float(state.ipac2) == pytest.approx(0.0, abs=0.0001)


class TestSWPPVPayoffs:
    """Test SWPPV payoff calculations."""

    def test_swppv_ied_payoff_is_zero(self):
        """Test that IED has zero payoff (no notional exchange)."""
        attrs = ContractAttributes(
            contract_id="SWAP001",
            contract_type=ContractType.SWPPV,
            contract_role=ContractRole.RPA,
            status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
            contract_deal_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
            initial_exchange_date=ActusDateTime(2024, 1, 15, 0, 0, 0),
            maturity_date=ActusDateTime(2026, 1, 15, 0, 0, 0),
            notional_principal=1000000.0,
            nominal_interest_rate=0.05,
            nominal_interest_rate_2=0.03,
            interest_payment_cycle="6M",
            rate_reset_cycle="3M",
            interest_calculation_base_anchor=ActusDateTime(2024, 1, 15, 0, 0, 0),
            rate_reset_anchor=ActusDateTime(2024, 1, 15, 0, 0, 0),
        )

        rf_obs = ConstantRiskFactorObserver(constant_value=0.04)
        swap = PlainVanillaSwapContract(attrs, rf_obs)

        state = swap.initialize_state()
        pof = swap.get_payoff_function(EventType.IED)

        payoff = pof.calculate_payoff(
            EventType.IED, state, attrs, attrs.initial_exchange_date, rf_obs
        )

        assert float(payoff) == pytest.approx(0.0, abs=0.01)

    def test_swppv_rr_payoff_is_zero(self):
        """Test that RR (Rate Reset) has zero payoff."""
        attrs = ContractAttributes(
            contract_id="SWAP001",
            contract_type=ContractType.SWPPV,
            contract_role=ContractRole.RPA,
            status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
            contract_deal_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
            initial_exchange_date=ActusDateTime(2024, 1, 15, 0, 0, 0),
            maturity_date=ActusDateTime(2026, 1, 15, 0, 0, 0),
            notional_principal=1000000.0,
            nominal_interest_rate=0.05,
            nominal_interest_rate_2=0.03,
            interest_payment_cycle="6M",
            rate_reset_cycle="3M",
            interest_calculation_base_anchor=ActusDateTime(2024, 1, 15, 0, 0, 0),
            rate_reset_anchor=ActusDateTime(2024, 1, 15, 0, 0, 0),
        )

        rf_obs = ConstantRiskFactorObserver(constant_value=0.04)
        swap = PlainVanillaSwapContract(attrs, rf_obs)

        state = swap.initialize_state()
        pof = swap.get_payoff_function(EventType.RR)

        rr_time = ActusDateTime(2024, 4, 15, 0, 0, 0)
        payoff = pof.calculate_payoff(EventType.RR, state, attrs, rr_time, rf_obs)

        assert float(payoff) == pytest.approx(0.0, abs=0.01)


class TestSWPPVAllPayoffFunctions:
    """Test all payoff functions for comprehensive coverage."""

    def test_swppv_ad_payoff_is_zero(self):
        """Test that AD (Analysis Date) has zero payoff."""
        attrs = ContractAttributes(
            contract_id="SWAP001",
            contract_type=ContractType.SWPPV,
            contract_role=ContractRole.RPA,
            status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
            maturity_date=ActusDateTime(2026, 1, 15, 0, 0, 0),
            notional_principal=1000000.0,
            nominal_interest_rate=0.05,
            nominal_interest_rate_2=0.03,
            interest_payment_cycle="6M",
            rate_reset_cycle="3M",
        )

        rf_obs = ConstantRiskFactorObserver(constant_value=0.04)
        swap = PlainVanillaSwapContract(attrs, rf_obs)
        state = swap.initialize_state()
        pof = swap.get_payoff_function(EventType.AD)

        payoff = pof.calculate_payoff(
            EventType.AD, state, attrs, ActusDateTime(2024, 3, 1, 0, 0, 0), rf_obs
        )
        assert float(payoff) == pytest.approx(0.0, abs=0.01)

    def test_swppv_pr_payoff_is_zero(self):
        """Test that PR (Principal Redemption) has zero payoff."""
        attrs = ContractAttributes(
            contract_id="SWAP001",
            contract_type=ContractType.SWPPV,
            contract_role=ContractRole.RPA,
            status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
            maturity_date=ActusDateTime(2026, 1, 15, 0, 0, 0),
            notional_principal=1000000.0,
            nominal_interest_rate=0.05,
            nominal_interest_rate_2=0.03,
            interest_payment_cycle="6M",
            rate_reset_cycle="3M",
        )

        rf_obs = ConstantRiskFactorObserver(constant_value=0.04)
        swap = PlainVanillaSwapContract(attrs, rf_obs)
        state = swap.initialize_state()
        pof = swap.get_payoff_function(EventType.PR)

        payoff = pof.calculate_payoff(
            EventType.PR, state, attrs, ActusDateTime(2024, 3, 1, 0, 0, 0), rf_obs
        )
        assert float(payoff) == pytest.approx(0.0, abs=0.01)

    def test_swppv_td_payoff_is_zero(self):
        """Test that TD (Termination Date) has zero payoff."""
        attrs = ContractAttributes(
            contract_id="SWAP001",
            contract_type=ContractType.SWPPV,
            contract_role=ContractRole.RPA,
            status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
            maturity_date=ActusDateTime(2026, 1, 15, 0, 0, 0),
            notional_principal=1000000.0,
            nominal_interest_rate=0.05,
            nominal_interest_rate_2=0.03,
            interest_payment_cycle="6M",
            rate_reset_cycle="3M",
        )

        rf_obs = ConstantRiskFactorObserver(constant_value=0.04)
        swap = PlainVanillaSwapContract(attrs, rf_obs)
        state = swap.initialize_state()
        pof = swap.get_payoff_function(EventType.TD)

        payoff = pof.calculate_payoff(
            EventType.TD, state, attrs, ActusDateTime(2024, 3, 1, 0, 0, 0), rf_obs
        )
        assert float(payoff) == pytest.approx(0.0, abs=0.01)

    def test_swppv_ce_payoff_is_zero(self):
        """Test that CE (Credit Event) has zero payoff."""
        attrs = ContractAttributes(
            contract_id="SWAP001",
            contract_type=ContractType.SWPPV,
            contract_role=ContractRole.RPA,
            status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
            maturity_date=ActusDateTime(2026, 1, 15, 0, 0, 0),
            notional_principal=1000000.0,
            nominal_interest_rate=0.05,
            nominal_interest_rate_2=0.03,
            interest_payment_cycle="6M",
            rate_reset_cycle="3M",
        )

        rf_obs = ConstantRiskFactorObserver(constant_value=0.04)
        swap = PlainVanillaSwapContract(attrs, rf_obs)
        state = swap.initialize_state()
        pof = swap.get_payoff_function(EventType.CE)

        payoff = pof.calculate_payoff(
            EventType.CE, state, attrs, ActusDateTime(2024, 3, 1, 0, 0, 0), rf_obs
        )
        assert float(payoff) == pytest.approx(0.0, abs=0.01)

    def test_swppv_ip_payoff_with_accrual(self):
        """Test IP payoff computes net of fixed and floating leg accruals.

        POF_IP accrues both legs from state.sd to event time, then nets them:
            net = (ipac1 + fixed_accrual) - (ipac2 + floating_accrual)
        With RPA role (receive fixed), payoff = +net.
        """
        attrs = ContractAttributes(
            contract_id="SWAP001",
            contract_type=ContractType.SWPPV,
            contract_role=ContractRole.RPA,  # Receive fixed
            status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
            maturity_date=ActusDateTime(2026, 1, 15, 0, 0, 0),
            notional_principal=1000000.0,
            nominal_interest_rate=0.05,  # Fixed leg: 5%
            nominal_interest_rate_2=0.03,  # Floating leg initial: 3%
            interest_payment_cycle="6M",
            rate_reset_cycle="3M",
            day_count_convention="A360",
        )

        rf_obs = ConstantRiskFactorObserver(constant_value=0.04)
        swap = PlainVanillaSwapContract(attrs, rf_obs)

        # State at IED with zero accruals, floating rate = 0.03
        state = swap.initialize_state()

        pof = swap.get_payoff_function(EventType.IP)
        # Payment at 6 months: 2024-01-01 to 2024-07-01 = 181 days / 360
        payoff = pof.calculate_payoff(
            EventType.IP, state, attrs, ActusDateTime(2024, 7, 1, 0, 0, 0), rf_obs
        )

        # Fixed leg accrual: 181/360 * 0.05 * 1M = 25138.89
        # Floating leg accrual: 181/360 * 0.03 * 1M = 15083.33
        # Net = 25138.89 - 15083.33 = 10055.56 (approx)
        # RPA receives fixed, so payoff > 0 when fixed > floating
        assert float(payoff) > 0.0
        assert float(payoff) == pytest.approx(10055.56, abs=100.0)


class TestSWPPVStateTransitions:
    """Test state transition functions."""

    def test_swppv_ied_state_transition(self):
        """Test IED state transition."""
        attrs = ContractAttributes(
            contract_id="SWAP001",
            contract_type=ContractType.SWPPV,
            contract_role=ContractRole.RPA,
            status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
            initial_exchange_date=ActusDateTime(2024, 1, 15, 0, 0, 0),
            maturity_date=ActusDateTime(2026, 1, 15, 0, 0, 0),
            notional_principal=1000000.0,
            nominal_interest_rate=0.05,
            nominal_interest_rate_2=0.03,
            interest_payment_cycle="6M",
            rate_reset_cycle="3M",
        )

        rf_obs = ConstantRiskFactorObserver(constant_value=0.04)
        swap = PlainVanillaSwapContract(attrs, rf_obs)

        state_pre = swap.initialize_state()
        stf = swap.get_state_transition_function(EventType.IED)

        state_post = stf.transition_state(
            EventType.IED, state_pre, attrs, attrs.initial_exchange_date, rf_obs
        )

        assert float(state_post.ipac) == pytest.approx(0.0, abs=0.01)
        assert float(state_post.ipnr) == pytest.approx(0.03, abs=0.01)

    def test_swppv_rr_state_transition(self):
        """Test RR state transition updates ipnr."""
        attrs = ContractAttributes(
            contract_id="SWAP001",
            contract_type=ContractType.SWPPV,
            contract_role=ContractRole.RPA,
            status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
            initial_exchange_date=ActusDateTime(2024, 1, 15, 0, 0, 0),
            maturity_date=ActusDateTime(2026, 1, 15, 0, 0, 0),
            notional_principal=1000000.0,
            nominal_interest_rate=0.05,
            nominal_interest_rate_2=0.03,
            interest_payment_cycle="6M",
            rate_reset_cycle="3M",
            rate_reset_market_object="LIBOR",
            rate_reset_multiplier=1.0,
            rate_reset_spread=0.005,  # 50bp spread
        )

        rf_obs = ConstantRiskFactorObserver(constant_value=0.04)  # Market rate 4%
        swap = PlainVanillaSwapContract(attrs, rf_obs)

        state_pre = swap.initialize_state()
        stf = swap.get_state_transition_function(EventType.RR)

        state_post = stf.transition_state(
            EventType.RR, state_pre, attrs, ActusDateTime(2024, 4, 15, 0, 0, 0), rf_obs
        )

        # New rate = 1.0 * 0.04 + 0.005 = 0.045
        assert float(state_post.ipnr) == pytest.approx(0.045, abs=0.001)

    def test_swppv_rr_with_cap(self):
        """Test RR state transition with rate cap."""
        attrs = ContractAttributes(
            contract_id="SWAP001",
            contract_type=ContractType.SWPPV,
            contract_role=ContractRole.RPA,
            status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
            maturity_date=ActusDateTime(2026, 1, 15, 0, 0, 0),
            notional_principal=1000000.0,
            nominal_interest_rate=0.05,
            nominal_interest_rate_2=0.03,
            interest_payment_cycle="6M",
            rate_reset_cycle="3M",
            rate_reset_cap=0.06,  # Cap at 6%
        )

        rf_obs = ConstantRiskFactorObserver(constant_value=0.08)  # Market rate 8% (above cap)
        swap = PlainVanillaSwapContract(attrs, rf_obs)

        state_pre = swap.initialize_state()
        stf = swap.get_state_transition_function(EventType.RR)

        state_post = stf.transition_state(
            EventType.RR, state_pre, attrs, ActusDateTime(2024, 4, 15, 0, 0, 0), rf_obs
        )

        # Rate should be capped at 0.06
        assert float(state_post.ipnr) == pytest.approx(0.06, abs=0.001)

    def test_swppv_rr_with_floor(self):
        """Test RR state transition with rate floor."""
        attrs = ContractAttributes(
            contract_id="SWAP001",
            contract_type=ContractType.SWPPV,
            contract_role=ContractRole.RPA,
            status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
            maturity_date=ActusDateTime(2026, 1, 15, 0, 0, 0),
            notional_principal=1000000.0,
            nominal_interest_rate=0.05,
            nominal_interest_rate_2=0.03,
            interest_payment_cycle="6M",
            rate_reset_cycle="3M",
            rate_reset_floor=0.02,  # Floor at 2%
        )

        rf_obs = ConstantRiskFactorObserver(constant_value=0.01)  # Market rate 1% (below floor)
        swap = PlainVanillaSwapContract(attrs, rf_obs)

        state_pre = swap.initialize_state()
        stf = swap.get_state_transition_function(EventType.RR)

        state_post = stf.transition_state(
            EventType.RR, state_pre, attrs, ActusDateTime(2024, 4, 15, 0, 0, 0), rf_obs
        )

        # Rate should be floored at 0.02
        assert float(state_post.ipnr) == pytest.approx(0.02, abs=0.001)

    def test_swppv_ip_state_transition(self):
        """Test IP state transition resets ipac."""
        attrs = ContractAttributes(
            contract_id="SWAP001",
            contract_type=ContractType.SWPPV,
            contract_role=ContractRole.RPA,
            status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
            maturity_date=ActusDateTime(2026, 1, 15, 0, 0, 0),
            notional_principal=1000000.0,
            nominal_interest_rate=0.05,
            nominal_interest_rate_2=0.03,
            interest_payment_cycle="6M",
            rate_reset_cycle="3M",
        )

        rf_obs = ConstantRiskFactorObserver(constant_value=0.04)
        swap = PlainVanillaSwapContract(attrs, rf_obs)

        # Create state with accrual
        state_pre = swap.initialize_state()
        from jactus.core import ContractState

        state_with_accrual = ContractState(
            tmd=state_pre.tmd,
            sd=state_pre.sd,
            nt=state_pre.nt,
            ipnr=state_pre.ipnr,
            ipac=jnp.array(0.025, dtype=jnp.float32),
            feac=state_pre.feac,
            nsc=state_pre.nsc,
            isc=state_pre.isc,
            prf=state_pre.prf,
        )

        stf = swap.get_state_transition_function(EventType.IP)
        state_post = stf.transition_state(
            EventType.IP, state_with_accrual, attrs, ActusDateTime(2024, 7, 15, 0, 0, 0), rf_obs
        )

        # Accrual should be reset to zero
        assert float(state_post.ipac) == pytest.approx(0.0, abs=0.01)

    def test_swppv_ad_state_transition(self):
        """Test AD state transition accrues both legs."""
        attrs = ContractAttributes(
            contract_id="SWAP001",
            contract_type=ContractType.SWPPV,
            contract_role=ContractRole.RPA,
            status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
            maturity_date=ActusDateTime(2026, 1, 15, 0, 0, 0),
            notional_principal=1000000.0,
            nominal_interest_rate=0.05,  # Fixed: 5%
            nominal_interest_rate_2=0.03,  # Floating: 3%
            interest_payment_cycle="6M",
            rate_reset_cycle="3M",
            day_count_convention="A360",
        )

        rf_obs = ConstantRiskFactorObserver(constant_value=0.04)
        swap = PlainVanillaSwapContract(attrs, rf_obs)

        state_pre = swap.initialize_state()
        stf = swap.get_state_transition_function(EventType.AD)

        state_post = stf.transition_state(
            EventType.AD, state_pre, attrs, ActusDateTime(2024, 3, 1, 0, 0, 0), rf_obs
        )

        # Status date should advance
        assert state_post.sd == ActusDateTime(2024, 3, 1, 0, 0, 0)
        # Fixed leg accrues: 60/360 * 0.05 * 1M = 8333.33
        assert float(state_post.ipac1) == pytest.approx(8333.33, abs=1.0)
        # Floating leg accrues: 60/360 * 0.03 * 1M = 5000.0
        assert float(state_post.ipac2) == pytest.approx(5000.0, abs=1.0)
        # ipac = R(CNTRL) × ipac1 = 1 × 8333.33 = 8333.33 (signed fixed leg accrual)
        assert float(state_post.ipac) == pytest.approx(8333.33, abs=1.0)
        # Notional unchanged
        assert float(state_post.nt) == pytest.approx(1000000.0)

    def test_swppv_pr_state_transition(self):
        """Test PR state transition doesn't change state."""
        attrs = ContractAttributes(
            contract_id="SWAP001",
            contract_type=ContractType.SWPPV,
            contract_role=ContractRole.RPA,
            status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
            maturity_date=ActusDateTime(2026, 1, 15, 0, 0, 0),
            notional_principal=1000000.0,
            nominal_interest_rate=0.05,
            nominal_interest_rate_2=0.03,
            interest_payment_cycle="6M",
            rate_reset_cycle="3M",
        )

        rf_obs = ConstantRiskFactorObserver(constant_value=0.04)
        swap = PlainVanillaSwapContract(attrs, rf_obs)

        state_pre = swap.initialize_state()
        stf = swap.get_state_transition_function(EventType.PR)

        state_post = stf.transition_state(
            EventType.PR, state_pre, attrs, ActusDateTime(2024, 3, 1, 0, 0, 0), rf_obs
        )

        # State should remain unchanged
        assert state_post == state_pre

    def test_swppv_td_state_transition(self):
        """Test TD state transition."""
        attrs = ContractAttributes(
            contract_id="SWAP001",
            contract_type=ContractType.SWPPV,
            contract_role=ContractRole.RPA,
            status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
            maturity_date=ActusDateTime(2026, 1, 15, 0, 0, 0),
            notional_principal=1000000.0,
            nominal_interest_rate=0.05,
            nominal_interest_rate_2=0.03,
            interest_payment_cycle="6M",
            rate_reset_cycle="3M",
        )

        rf_obs = ConstantRiskFactorObserver(constant_value=0.04)
        swap = PlainVanillaSwapContract(attrs, rf_obs)

        state_pre = swap.initialize_state()
        stf = swap.get_state_transition_function(EventType.TD)

        td_time = ActusDateTime(2025, 6, 1, 0, 0, 0)
        state_post = stf.transition_state(EventType.TD, state_pre, attrs, td_time, rf_obs)

        # tmd should be updated to termination time
        assert state_post.tmd == td_time

    def test_swppv_ce_state_transition(self):
        """Test CE state transition doesn't change state."""
        attrs = ContractAttributes(
            contract_id="SWAP001",
            contract_type=ContractType.SWPPV,
            contract_role=ContractRole.RPA,
            status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
            maturity_date=ActusDateTime(2026, 1, 15, 0, 0, 0),
            notional_principal=1000000.0,
            nominal_interest_rate=0.05,
            nominal_interest_rate_2=0.03,
            interest_payment_cycle="6M",
            rate_reset_cycle="3M",
        )

        rf_obs = ConstantRiskFactorObserver(constant_value=0.04)
        swap = PlainVanillaSwapContract(attrs, rf_obs)

        state_pre = swap.initialize_state()
        stf = swap.get_state_transition_function(EventType.CE)

        state_post = stf.transition_state(
            EventType.CE, state_pre, attrs, ActusDateTime(2024, 3, 1, 0, 0, 0), rf_obs
        )

        # State should remain unchanged
        assert state_post == state_pre


class TestSWPPVFactory:
    """Test SWPPV factory creation."""

    def test_create_swppv_via_factory(self):
        """Test creating SWPPV contract using create_contract factory."""
        attrs = ContractAttributes(
            contract_id="SWAP001",
            contract_type=ContractType.SWPPV,
            contract_role=ContractRole.RPA,
            status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
            contract_deal_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
            initial_exchange_date=ActusDateTime(2024, 1, 15, 0, 0, 0),
            maturity_date=ActusDateTime(2026, 1, 15, 0, 0, 0),
            notional_principal=1000000.0,
            nominal_interest_rate=0.05,
            nominal_interest_rate_2=0.03,
            interest_payment_cycle="6M",
            rate_reset_cycle="3M",
            interest_calculation_base_anchor=ActusDateTime(2024, 1, 15, 0, 0, 0),
            rate_reset_anchor=ActusDateTime(2024, 1, 15, 0, 0, 0),
        )

        rf_obs = ConstantRiskFactorObserver(constant_value=0.04)

        # Create via factory
        swap = create_contract(attrs, rf_obs)

        # Should be a PlainVanillaSwapContract instance
        assert isinstance(swap, PlainVanillaSwapContract)
        assert swap.attributes.contract_type == ContractType.SWPPV
