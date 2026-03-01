"""Tests for the unified portfolio simulation API.

Verifies that simulate_portfolio correctly groups mixed contract types,
dispatches to batch kernels, and falls back to scalar path for
unsupported types.
"""

import jax.numpy as jnp
import pytest

from jactus.contracts.portfolio import (
    BATCH_SUPPORTED_TYPES,
    simulate_portfolio,
)
from jactus.core import (
    ActusDateTime,
    ContractAttributes,
    ContractRole,
    ContractType,
    DayCountConvention,
)
from jactus.observers import ConstantRiskFactorObserver, DictRiskFactorObserver

ATOL = 1.0


# ============================================================================
# Fixtures
# ============================================================================


def _make_pam(notional: float = 100_000.0) -> ContractAttributes:
    return ContractAttributes(
        contract_id="PAM-001",
        contract_type=ContractType.PAM,
        contract_role=ContractRole.RPA,
        status_date=ActusDateTime(2024, 1, 1),
        initial_exchange_date=ActusDateTime(2024, 1, 15),
        maturity_date=ActusDateTime(2027, 1, 15),
        currency="USD",
        notional_principal=notional,
        nominal_interest_rate=0.05,
        day_count_convention=DayCountConvention.A360,
        interest_payment_cycle="1Y",
    )


def _make_lam(notional: float = 100_000.0) -> ContractAttributes:
    return ContractAttributes(
        contract_id="LAM-001",
        contract_type=ContractType.LAM,
        contract_role=ContractRole.RPA,
        status_date=ActusDateTime(2024, 1, 1),
        initial_exchange_date=ActusDateTime(2024, 1, 15),
        maturity_date=ActusDateTime(2027, 1, 15),
        currency="USD",
        notional_principal=notional,
        nominal_interest_rate=0.05,
        day_count_convention=DayCountConvention.A360,
        interest_payment_cycle="1Y",
        principal_redemption_cycle="1Y",
    )


def _make_csh(notional: float = 50_000.0) -> ContractAttributes:
    return ContractAttributes(
        contract_id="CSH-001",
        contract_type=ContractType.CSH,
        contract_role=ContractRole.RPA,
        status_date=ActusDateTime(2024, 1, 1),
        currency="USD",
        notional_principal=notional,
    )


def _make_stk() -> ContractAttributes:
    return ContractAttributes(
        contract_id="STK-001",
        contract_type=ContractType.STK,
        contract_role=ContractRole.RPA,
        status_date=ActusDateTime(2024, 1, 1),
        currency="USD",
        purchase_date=ActusDateTime(2024, 1, 15),
        termination_date=ActusDateTime(2025, 1, 15),
        price_at_purchase_date=100.0,
        price_at_termination_date=120.0,
    )


def _make_fxout() -> ContractAttributes:
    return ContractAttributes(
        contract_id="FXOUT-001",
        contract_type=ContractType.FXOUT,
        contract_role=ContractRole.RPA,
        status_date=ActusDateTime(2024, 1, 1),
        maturity_date=ActusDateTime(2025, 1, 15),
        currency="USD",
        currency_2="EUR",
        notional_principal=100_000.0,
        notional_principal_2=90_000.0,
        delivery_settlement="D",
    )


def _make_optns() -> ContractAttributes:
    return ContractAttributes(
        contract_id="OPTNS-001",
        contract_type=ContractType.OPTNS,
        contract_role=ContractRole.BUY,
        status_date=ActusDateTime(2024, 1, 1),
        maturity_date=ActusDateTime(2025, 1, 15),
        currency="USD",
        notional_principal=100.0,
        option_strike_1=100.0,
        option_type="C",
        option_exercise_type="E",
        purchase_date=ActusDateTime(2024, 1, 15),
        price_at_purchase_date=10.0,
        contract_structure="SPX",
    )


def _make_ann(notional: float = 100_000.0) -> ContractAttributes:
    return ContractAttributes(
        contract_id="ANN-001",
        contract_type=ContractType.ANN,
        contract_role=ContractRole.RPA,
        status_date=ActusDateTime(2024, 1, 1),
        initial_exchange_date=ActusDateTime(2024, 1, 15),
        maturity_date=ActusDateTime(2029, 1, 15),
        currency="USD",
        notional_principal=notional,
        nominal_interest_rate=0.05,
        day_count_convention=DayCountConvention.A360,
        interest_payment_cycle="1Y",
        principal_redemption_cycle="1Y",
    )


def _make_swppv() -> ContractAttributes:
    return ContractAttributes(
        contract_id="SWPPV-001",
        contract_type=ContractType.SWPPV,
        contract_role=ContractRole.RFL,
        status_date=ActusDateTime(2024, 1, 1),
        initial_exchange_date=ActusDateTime(2024, 1, 15),
        maturity_date=ActusDateTime(2027, 1, 15),
        currency="USD",
        notional_principal=1_000_000.0,
        nominal_interest_rate=0.03,
        nominal_interest_rate_2=0.04,
        day_count_convention=DayCountConvention.A360,
        interest_payment_cycle="6M",
    )


# ============================================================================
# Tests
# ============================================================================


class TestPortfolioBasics:
    """Basic portfolio simulation tests."""

    def test_empty_portfolio(self):
        """Empty portfolio should return empty results."""
        result = simulate_portfolio([])
        assert result["num_contracts"] == 0
        assert result["total_cashflows"].shape == (0,)

    def test_single_pam(self):
        """Single PAM contract should work."""
        rf_obs = ConstantRiskFactorObserver(0.0)
        result = simulate_portfolio([(_make_pam(), rf_obs)])
        assert result["num_contracts"] == 1
        assert result["batch_contracts"] == 1
        assert result["fallback_contracts"] == 0
        assert result["total_cashflows"].shape == (1,)
        assert float(result["total_cashflows"][0]) != 0.0

    def test_single_csh(self):
        """Single CSH contract — cashflows should be zero."""
        rf_obs = ConstantRiskFactorObserver(0.0)
        result = simulate_portfolio([(_make_csh(), rf_obs)])
        assert result["num_contracts"] == 1
        assert abs(float(result["total_cashflows"][0])) < ATOL


class TestMixedPortfolio:
    """Tests for portfolios with mixed contract types."""

    def test_pam_and_lam(self):
        """Mixed PAM + LAM portfolio."""
        rf_obs = ConstantRiskFactorObserver(0.0)
        contracts = [
            (_make_pam(), rf_obs),
            (_make_lam(), rf_obs),
            (_make_pam(notional=50_000.0), rf_obs),
        ]
        result = simulate_portfolio(contracts)
        assert result["num_contracts"] == 3
        assert result["batch_contracts"] == 3
        assert result["fallback_contracts"] == 0
        assert result["total_cashflows"].shape == (3,)
        assert ContractType.PAM in result["types_used"]
        assert ContractType.LAM in result["types_used"]

    def test_many_types(self):
        """Portfolio with many different types."""
        rf_obs = ConstantRiskFactorObserver(0.0)
        spx_obs = DictRiskFactorObserver({"SPX": 110.0})
        contracts = [
            (_make_pam(), rf_obs),
            (_make_lam(), rf_obs),
            (_make_csh(), rf_obs),
            (_make_stk(), rf_obs),
            (_make_fxout(), rf_obs),
            (_make_optns(), spx_obs),
            (_make_ann(), rf_obs),
            (_make_swppv(), rf_obs),
        ]
        result = simulate_portfolio(contracts)
        assert result["num_contracts"] == 8
        assert result["batch_contracts"] == 8
        assert result["fallback_contracts"] == 0
        assert result["total_cashflows"].shape == (8,)
        assert len(result["types_used"]) == 8

    def test_order_preserved(self):
        """Results should be in the same order as input contracts."""
        rf_obs = ConstantRiskFactorObserver(0.0)
        contracts = [
            (_make_pam(notional=100_000.0), rf_obs),
            (_make_csh(), rf_obs),
            (_make_pam(notional=200_000.0), rf_obs),
        ]
        result = simulate_portfolio(contracts)

        # CSH should have ~zero cashflow (index 1)
        assert abs(float(result["total_cashflows"][1])) < ATOL

        # PAM with 200k should have roughly 2x the cashflow of 100k PAM
        cf_100k = float(result["total_cashflows"][0])
        cf_200k = float(result["total_cashflows"][2])
        assert abs(cf_200k / cf_100k - 2.0) < 0.01


class TestEquivalenceWithIndividual:
    """Verify portfolio results match individual simulations."""

    def test_pam_portfolio_matches_individual(self):
        """PAM portfolio cashflows should match individual simulations."""
        rf_obs = ConstantRiskFactorObserver(0.0)
        contracts = [
            (_make_pam(notional=100_000.0), rf_obs),
            (_make_pam(notional=50_000.0), rf_obs),
        ]

        # Portfolio
        port_result = simulate_portfolio(contracts)

        # Individual via the same array path
        from jactus.contracts.pam_array import (
            precompute_pam_arrays,
            simulate_pam_array,
        )

        individual_totals = []
        for attrs, obs in contracts:
            arrays = precompute_pam_arrays(attrs, obs)
            _, payoffs = simulate_pam_array(*arrays)
            individual_totals.append(float(jnp.sum(payoffs)))

        for i in range(len(contracts)):
            assert abs(float(port_result["total_cashflows"][i]) - individual_totals[i]) <= ATOL

    def test_mixed_portfolio_matches_individual(self):
        """Mixed portfolio cashflows should match individual scalar simulations."""
        rf_obs = ConstantRiskFactorObserver(0.0)
        contracts = [
            (_make_pam(), rf_obs),
            (_make_lam(), rf_obs),
            (_make_csh(), rf_obs),
        ]

        # Portfolio
        port_result = simulate_portfolio(contracts)

        # Individual via scalar path
        from jactus.contracts import create_contract

        for i, (attrs, obs) in enumerate(contracts):
            contract = create_contract(attrs, obs)
            py_result = contract.simulate()
            py_total = sum(float(e.payoff) for e in py_result.events)
            assert abs(float(port_result["total_cashflows"][i]) - py_total) <= ATOL


class TestFallbackPath:
    """Tests for the scalar Python fallback path."""

    def test_clm_uses_fallback(self):
        """CLM should use the scalar fallback path."""
        attrs = ContractAttributes(
            contract_id="CLM-001",
            contract_type=ContractType.CLM,
            contract_role=ContractRole.RPA,
            status_date=ActusDateTime(2024, 1, 1),
            initial_exchange_date=ActusDateTime(2024, 1, 15),
            maturity_date=ActusDateTime(2025, 1, 15),
            currency="USD",
            notional_principal=100_000.0,
            nominal_interest_rate=0.05,
            day_count_convention=DayCountConvention.A360,
            interest_payment_cycle="1M",
        )
        rf_obs = ConstantRiskFactorObserver(0.0)
        result = simulate_portfolio([(attrs, rf_obs)])
        assert result["fallback_contracts"] == 1
        assert result["batch_contracts"] == 0
        assert float(result["total_cashflows"][0]) != 0.0

    def test_mixed_batch_and_fallback(self):
        """Portfolio with both batch and fallback types."""
        clm_attrs = ContractAttributes(
            contract_id="CLM-001",
            contract_type=ContractType.CLM,
            contract_role=ContractRole.RPA,
            status_date=ActusDateTime(2024, 1, 1),
            initial_exchange_date=ActusDateTime(2024, 1, 15),
            maturity_date=ActusDateTime(2025, 1, 15),
            currency="USD",
            notional_principal=100_000.0,
            nominal_interest_rate=0.05,
            day_count_convention=DayCountConvention.A360,
            interest_payment_cycle="1M",
        )
        rf_obs = ConstantRiskFactorObserver(0.0)
        contracts = [
            (_make_pam(), rf_obs),
            (clm_attrs, rf_obs),
            (_make_csh(), rf_obs),
        ]
        result = simulate_portfolio(contracts)
        assert result["num_contracts"] == 3
        assert result["batch_contracts"] == 2
        assert result["fallback_contracts"] == 1
        assert ContractType.PAM in result["per_type_results"]
        assert ContractType.CSH in result["per_type_results"]
        assert ContractType.CLM not in result["per_type_results"]


class TestPerTypeResults:
    """Verify per-type results are accessible."""

    def test_per_type_results_structure(self):
        """Per-type results should contain expected keys."""
        rf_obs = ConstantRiskFactorObserver(0.0)
        contracts = [
            (_make_pam(), rf_obs),
            (_make_pam(notional=50_000.0), rf_obs),
        ]
        result = simulate_portfolio(contracts)
        assert ContractType.PAM in result["per_type_results"]
        pam_result = result["per_type_results"][ContractType.PAM]
        assert "total_cashflows" in pam_result
        assert "payoffs" in pam_result
        assert pam_result["num_contracts"] == 2


class TestBatchSupportedTypes:
    """Verify the BATCH_SUPPORTED_TYPES constant."""

    def test_all_batch_types_listed(self):
        """All 12 batch-supported types should be in the constant."""
        expected = {
            ContractType.PAM,
            ContractType.LAM,
            ContractType.NAM,
            ContractType.ANN,
            ContractType.LAX,
            ContractType.CSH,
            ContractType.STK,
            ContractType.COM,
            ContractType.FXOUT,
            ContractType.FUTUR,
            ContractType.OPTNS,
            ContractType.SWPPV,
        }
        assert BATCH_SUPPORTED_TYPES == expected
