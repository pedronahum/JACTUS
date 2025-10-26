"""Tests for financial mathematics utilities.

Tests the implementation of financial calculations including contract role signs,
annuity calculations, discount factors, and present value.
"""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from jactus.core.time import ActusDateTime
from jactus.core.types import ContractRole, DayCountConvention
from jactus.utilities.math import (
    annuity_amount,
    annuity_amount_vectorized,
    compound_factor,
    compound_factor_vectorized,
    contract_role_sign,
    contract_role_sign_vectorized,
    discount_factor,
    discount_factor_vectorized,
    present_value,
    present_value_vectorized,
)


class TestContractRoleSign:
    """Test contract role sign function."""

    def test_receiving_party_a(self):
        """RPA (Receiving Party A) has sign +1."""
        assert contract_role_sign(ContractRole.RPA) == 1

    def test_real_position_lender(self):
        """RPL (Real Position Lender) has sign -1."""
        assert contract_role_sign(ContractRole.RPL) == -1

    def test_long(self):
        """Long positions have sign +1."""
        assert contract_role_sign(ContractRole.LG) == 1

    def test_short(self):
        """Short positions have sign -1."""
        assert contract_role_sign(ContractRole.ST) == -1

    def test_receiving_fixed_leg(self):
        """RFL (Receiving Fixed Leg) has sign +1."""
        assert contract_role_sign(ContractRole.RFL) == 1

    def test_paying_fixed_leg(self):
        """PFL (Paying Fixed Leg) has sign -1."""
        assert contract_role_sign(ContractRole.PFL) == -1

    def test_buyer(self):
        """BUY (Buyer) has sign +1."""
        assert contract_role_sign(ContractRole.BUY) == 1

    def test_seller(self):
        """SEL (Seller) has sign -1."""
        assert contract_role_sign(ContractRole.SEL) == -1

    def test_guarantor(self):
        """GUA (Guarantor) has sign -1."""
        assert contract_role_sign(ContractRole.GUA) == -1


class TestContractRoleSignVectorized:
    """Test vectorized contract role sign function."""

    def test_single_role(self):
        """Vectorized works with single role."""
        roles = jnp.array([0])  # RPA
        signs = contract_role_sign_vectorized(roles)
        assert signs[0] == 1

    def test_multiple_roles(self):
        """Vectorized works with multiple roles."""
        # RPA, RPL, CLG
        roles = jnp.array([0, 1, 2])
        signs = contract_role_sign_vectorized(roles)

        assert signs[0] == 1  # RPA
        assert signs[1] == -1  # RPL
        assert signs[2] == 1  # CLG

    def test_all_roles(self):
        """Test all 15 contract roles."""
        # All role indices 0-14
        roles = jnp.arange(15)
        signs = contract_role_sign_vectorized(roles)

        # RPA, RPL, LG, ST, BUY, SEL, RFL, PFL, COL, CNO, GUA, OBL, UDL, UDLP, UDLM
        expected = jnp.array([1, -1, 1, -1, 1, -1, 1, -1, 1, 1, -1, 1, 1, 1, -1])
        assert jnp.array_equal(signs, expected)


class TestAnnuityAmount:
    """Test annuity amount calculation."""

    def test_simple_annuity(self):
        """Calculate annuity for simple case."""
        tenor = ActusDateTime(2024, 1, 1, 0, 0, 0)
        maturity = ActusDateTime(2025, 1, 1, 0, 0, 0)

        # $100,000 at 5% annual, 12 monthly payments
        amount = annuity_amount(
            100000.0,
            0.05 / 12,  # Monthly rate
            tenor,
            maturity,
            12,
            DayCountConvention.A360,
        )

        # Expected: ~$8,560.75 per month
        assert abs(amount - 8560.75) < 1.0

    def test_zero_rate(self):
        """Zero rate gives equal payments."""
        tenor = ActusDateTime(2024, 1, 1, 0, 0, 0)
        maturity = ActusDateTime(2025, 1, 1, 0, 0, 0)

        amount = annuity_amount(12000.0, 0.0, tenor, maturity, 12, DayCountConvention.A360)

        # $12,000 / 12 months = $1,000/month
        assert abs(amount - 1000.0) < 0.01

    def test_zero_periods(self):
        """Zero periods gives zero."""
        tenor = ActusDateTime(2024, 1, 1, 0, 0, 0)
        maturity = ActusDateTime(2025, 1, 1, 0, 0, 0)

        amount = annuity_amount(100000.0, 0.05, tenor, maturity, 0, DayCountConvention.A360)
        assert amount == 0.0

    def test_single_period(self):
        """Single period returns notional + interest."""
        tenor = ActusDateTime(2024, 1, 1, 0, 0, 0)
        maturity = ActusDateTime(2025, 1, 1, 0, 0, 0)

        amount = annuity_amount(100000.0, 0.05, tenor, maturity, 1, DayCountConvention.A360)

        # Single payment = principal + interest
        assert abs(amount - 105000.0) < 1.0


class TestAnnuityAmountVectorized:
    """Test vectorized annuity calculation."""

    def test_multiple_annuities(self):
        """Calculate multiple annuities at once."""
        notionals = jnp.array([100000.0, 200000.0, 50000.0])
        rates = jnp.array([0.05 / 12, 0.04 / 12, 0.06 / 12])
        periods = jnp.array([12, 24, 6])

        amounts = annuity_amount_vectorized(notionals, rates, periods)

        # Check reasonable values
        assert amounts[0] > 8500 and amounts[0] < 8600  # ~$8,560
        assert amounts[1] > 8600 and amounts[1] < 8700  # ~$8,685
        assert amounts[2] > 8400 and amounts[2] < 8500  # ~$8,480

    def test_zero_rates_vectorized(self):
        """Handle zero rates in vectorized calculation."""
        notionals = jnp.array([12000.0, 24000.0])
        rates = jnp.array([0.0, 0.0])
        periods = jnp.array([12, 24])

        amounts = annuity_amount_vectorized(notionals, rates, periods)

        assert abs(amounts[0] - 1000.0) < 0.01  # 12000/12
        assert abs(amounts[1] - 1000.0) < 0.01  # 24000/24


class TestDiscountFactor:
    """Test discount factor calculation."""

    def test_six_months(self):
        """Discount factor for 6 months."""
        start = ActusDateTime(2024, 1, 1, 0, 0, 0)
        end = ActusDateTime(2024, 7, 1, 0, 0, 0)

        df = discount_factor(0.05, start, end, DayCountConvention.AA)

        # DF = 1 / (1 + 0.05 * 0.5) = 1 / 1.025 ≈ 0.9756
        assert abs(df - 0.9756) < 0.001

    def test_one_year(self):
        """Discount factor for 1 year."""
        start = ActusDateTime(2024, 1, 1, 0, 0, 0)
        end = ActusDateTime(2025, 1, 1, 0, 0, 0)

        df = discount_factor(0.05, start, end, DayCountConvention.AA)

        # DF = 1 / (1 + 0.05 * 1.0) = 1 / 1.05 ≈ 0.9524
        assert abs(df - 0.9524) < 0.001

    def test_zero_rate(self):
        """Zero rate gives discount factor of 1."""
        start = ActusDateTime(2024, 1, 1, 0, 0, 0)
        end = ActusDateTime(2025, 1, 1, 0, 0, 0)

        df = discount_factor(0.0, start, end, DayCountConvention.AA)
        assert abs(df - 1.0) < 0.0001

    def test_same_date(self):
        """Same date gives discount factor of 1."""
        date = ActusDateTime(2024, 1, 1, 0, 0, 0)

        df = discount_factor(0.05, date, date, DayCountConvention.AA)
        assert abs(df - 1.0) < 0.0001


class TestDiscountFactorVectorized:
    """Test vectorized discount factor calculation."""

    def test_multiple_discount_factors(self):
        """Calculate multiple discount factors."""
        rates = jnp.array([0.05, 0.04, 0.06])
        year_fractions = jnp.array([0.5, 1.0, 0.25])

        dfs = discount_factor_vectorized(rates, year_fractions)

        # Check values
        assert abs(dfs[0] - 0.9756) < 0.001  # 1/(1 + 0.05*0.5)
        assert abs(dfs[1] - 0.9615) < 0.001  # 1/(1 + 0.04*1.0)
        assert abs(dfs[2] - 0.9852) < 0.001  # 1/(1 + 0.06*0.25)


class TestCompoundFactor:
    """Test compound factor calculation."""

    def test_annual_compounding(self):
        """Annual compounding for one year."""
        start = ActusDateTime(2024, 1, 1, 0, 0, 0)
        end = ActusDateTime(2025, 1, 1, 0, 0, 0)

        cf = compound_factor(0.05, start, end, DayCountConvention.AA, 1)

        # (1 + 0.05)^1 = 1.05
        assert abs(cf - 1.05) < 0.001

    def test_monthly_compounding(self):
        """Monthly compounding for one year."""
        start = ActusDateTime(2024, 1, 1, 0, 0, 0)
        end = ActusDateTime(2025, 1, 1, 0, 0, 0)

        cf = compound_factor(0.05, start, end, DayCountConvention.AA, 12)

        # (1 + 0.05/12)^12 ≈ 1.05116
        assert abs(cf - 1.05116) < 0.001

    def test_continuous_compounding(self):
        """Continuous compounding (frequency = 0)."""
        start = ActusDateTime(2024, 1, 1, 0, 0, 0)
        end = ActusDateTime(2025, 1, 1, 0, 0, 0)

        cf = compound_factor(0.05, start, end, DayCountConvention.AA, 0)

        # e^0.05 ≈ 1.05127
        assert abs(cf - 1.05127) < 0.001

    def test_six_months_quarterly(self):
        """Quarterly compounding for 6 months."""
        start = ActusDateTime(2024, 1, 1, 0, 0, 0)
        end = ActusDateTime(2024, 7, 1, 0, 0, 0)

        cf = compound_factor(0.05, start, end, DayCountConvention.AA, 4)

        # (1 + 0.05/4)^(4*0.5) ≈ 1.02516
        assert abs(cf - 1.02516) < 0.001


class TestCompoundFactorVectorized:
    """Test vectorized compound factor calculation."""

    def test_multiple_compound_factors(self):
        """Calculate multiple compound factors."""
        rates = jnp.array([0.05, 0.04])
        year_fractions = jnp.array([1.0, 1.0])
        frequencies = jnp.array([1, 12])

        cfs = compound_factor_vectorized(rates, year_fractions, frequencies)

        assert abs(cfs[0] - 1.05) < 0.001  # Annual
        assert abs(cfs[1] - 1.04074) < 0.001  # Monthly


class TestPresentValue:
    """Test present value calculation."""

    def test_single_cash_flow(self):
        """PV of single future cash flow."""
        cfs = [100.0]
        dates = [ActusDateTime(2025, 1, 1, 0, 0, 0)]
        val_date = ActusDateTime(2024, 1, 1, 0, 0, 0)

        pv = present_value(cfs, dates, val_date, 0.05, DayCountConvention.AA)

        # 100 / (1 + 0.05*1) ≈ 95.24
        assert abs(pv - 95.24) < 0.1

    def test_multiple_cash_flows(self):
        """PV of multiple cash flows."""
        cfs = [100.0, 100.0, 100.0]
        dates = [
            ActusDateTime(2025, 1, 1, 0, 0, 0),
            ActusDateTime(2026, 1, 1, 0, 0, 0),
            ActusDateTime(2027, 1, 1, 0, 0, 0),
        ]
        val_date = ActusDateTime(2024, 1, 1, 0, 0, 0)

        pv = present_value(cfs, dates, val_date, 0.05, DayCountConvention.AA)

        # Sum of discounted cash flows
        # 100/1.05 + 100/1.10 + 100/1.15 ≈ 272.3
        assert abs(pv - 272.3) < 1.0

    def test_cash_flow_at_valuation_date(self):
        """Cash flow at valuation date."""
        cfs = [100.0]
        dates = [ActusDateTime(2024, 1, 1, 0, 0, 0)]
        val_date = ActusDateTime(2024, 1, 1, 0, 0, 0)

        pv = present_value(cfs, dates, val_date, 0.05, DayCountConvention.AA)

        # No discounting needed
        assert abs(pv - 100.0) < 0.01

    def test_mismatched_lengths(self):
        """Raise error for mismatched lengths."""
        cfs = [100.0, 200.0]
        dates = [ActusDateTime(2025, 1, 1, 0, 0, 0)]
        val_date = ActusDateTime(2024, 1, 1, 0, 0, 0)

        with pytest.raises(ValueError, match="same length"):
            present_value(cfs, dates, val_date, 0.05, DayCountConvention.AA)


class TestPresentValueVectorized:
    """Test vectorized present value calculation."""

    def test_multiple_cash_flows_vectorized(self):
        """Vectorized PV calculation."""
        cfs = jnp.array([100.0, 100.0, 100.0])
        yfs = jnp.array([1.0, 2.0, 3.0])

        pv = present_value_vectorized(cfs, yfs, 0.05)

        # 100/1.05 + 100/1.10 + 100/1.15 ≈ 272.3
        assert abs(pv - 272.3) < 1.0

    def test_zero_year_fractions(self):
        """Handle zero year fractions."""
        cfs = jnp.array([100.0, 100.0])
        yfs = jnp.array([0.0, 1.0])

        pv = present_value_vectorized(cfs, yfs, 0.05)

        # 100/1.0 + 100/1.05 ≈ 195.24
        assert abs(pv - 195.24) < 0.1


class TestEdgeCases:
    """Test edge cases for financial math functions."""

    def test_very_high_rate(self):
        """Handle very high interest rates."""
        start = ActusDateTime(2024, 1, 1, 0, 0, 0)
        end = ActusDateTime(2025, 1, 1, 0, 0, 0)

        df = discount_factor(2.0, start, end, DayCountConvention.AA)  # 200%

        # DF = 1 / (1 + 2.0) = 0.333...
        assert abs(df - 0.3333) < 0.001

    def test_very_small_rate(self):
        """Handle very small rates."""
        tenor = ActusDateTime(2024, 1, 1, 0, 0, 0)
        maturity = ActusDateTime(2025, 1, 1, 0, 0, 0)

        amount = annuity_amount(12000.0, 1e-10, tenor, maturity, 12, DayCountConvention.A360)

        # Should fall back to simple division
        assert abs(amount - 1000.0) < 0.01

    def test_negative_cash_flows(self):
        """Handle negative cash flows in PV."""
        cfs = [-100.0, 200.0, -50.0]
        dates = [
            ActusDateTime(2025, 1, 1, 0, 0, 0),
            ActusDateTime(2026, 1, 1, 0, 0, 0),
            ActusDateTime(2027, 1, 1, 0, 0, 0),
        ]
        val_date = ActusDateTime(2024, 1, 1, 0, 0, 0)

        pv = present_value(cfs, dates, val_date, 0.05, DayCountConvention.AA)

        # Sum of discounted cash flows (mix of positive/negative)
        # Should be positive overall
        assert pv > 0
