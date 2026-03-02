"""Equivalence tests for array-mode LAM simulation.

Runs identical contracts through both the Python path (LinearAmortizerContract)
and the array-mode path (simulate_lam_array), asserting matching results.
Tolerance matches the ACTUS cross-validation standard (atol=1.0).
"""

import jax
import jax.numpy as jnp

from jactus.contracts.lam import LinearAmortizerContract
from jactus.contracts.lam_array import (
    precompute_lam_arrays,
    simulate_lam_array,
    simulate_lam_array_jit,
    simulate_lam_portfolio,
)
from jactus.core import (
    ActusDateTime,
    ContractAttributes,
    ContractRole,
    ContractType,
    DayCountConvention,
)
from jactus.observers import ConstantRiskFactorObserver, TimeSeriesRiskFactorObserver

ATOL = 1.0


# ============================================================================
# Fixtures
# ============================================================================


def _make_fixed_lam_attrs(
    notional: float = 100_000.0,
    rate: float = 0.05,
    years: int = 5,
    prnxt: float = 20_000.0,
    dcc: DayCountConvention = DayCountConvention.A360,
    ip_cycle: str = "1Y",
    pr_cycle: str = "1Y",
    role: ContractRole = ContractRole.RPA,
) -> ContractAttributes:
    """Create attributes for a simple fixed-rate LAM loan."""
    return ContractAttributes(
        contract_id="LAM-FIXED",
        contract_type=ContractType.LAM,
        contract_role=role,
        status_date=ActusDateTime(2024, 1, 1),
        initial_exchange_date=ActusDateTime(2024, 1, 15),
        maturity_date=ActusDateTime(2024 + years, 1, 15),
        currency="USD",
        notional_principal=notional,
        nominal_interest_rate=rate,
        day_count_convention=dcc,
        interest_payment_cycle=ip_cycle,
        principal_redemption_cycle=pr_cycle,
        next_principal_redemption_amount=prnxt,
    )


def _make_auto_prnxt_attrs() -> ContractAttributes:
    """LAM with auto-calculated PRNXT (not explicitly specified)."""
    return ContractAttributes(
        contract_id="LAM-AUTO",
        contract_type=ContractType.LAM,
        contract_role=ContractRole.RPA,
        status_date=ActusDateTime(2024, 1, 1),
        initial_exchange_date=ActusDateTime(2024, 1, 15),
        maturity_date=ActusDateTime(2029, 1, 15),
        currency="USD",
        notional_principal=100_000.0,
        nominal_interest_rate=0.05,
        day_count_convention=DayCountConvention.A360,
        interest_payment_cycle="1Y",
        principal_redemption_cycle="1Y",
    )


def _make_variable_rate_lam_attrs() -> ContractAttributes:
    """LAM with variable rate (rate resets)."""
    return ContractAttributes(
        contract_id="LAM-VARIABLE",
        contract_type=ContractType.LAM,
        contract_role=ContractRole.RPA,
        status_date=ActusDateTime(2024, 1, 1),
        initial_exchange_date=ActusDateTime(2024, 1, 15),
        maturity_date=ActusDateTime(2027, 1, 15),
        currency="USD",
        notional_principal=120_000.0,
        nominal_interest_rate=0.04,
        day_count_convention=DayCountConvention.A360,
        interest_payment_cycle="6M",
        principal_redemption_cycle="6M",
        next_principal_redemption_amount=20_000.0,
        rate_reset_cycle="6M",
        rate_reset_spread=0.01,
        rate_reset_multiplier=1.0,
        rate_reset_market_object="LIBOR-6M",
    )


def _make_midlife_lam_attrs() -> ContractAttributes:
    """Mid-life LAM (IED < SD)."""
    return ContractAttributes(
        contract_id="LAM-MIDLIFE",
        contract_type=ContractType.LAM,
        contract_role=ContractRole.RPA,
        status_date=ActusDateTime(2026, 1, 1),
        initial_exchange_date=ActusDateTime(2024, 1, 15),
        maturity_date=ActusDateTime(2029, 1, 15),
        currency="USD",
        notional_principal=100_000.0,
        nominal_interest_rate=0.05,
        day_count_convention=DayCountConvention.A360,
        interest_payment_cycle="1Y",
        principal_redemption_cycle="1Y",
        next_principal_redemption_amount=20_000.0,
    )


def _make_monthly_lam_attrs() -> ContractAttributes:
    """LAM with monthly payments."""
    return ContractAttributes(
        contract_id="LAM-MONTHLY",
        contract_type=ContractType.LAM,
        contract_role=ContractRole.RPA,
        status_date=ActusDateTime(2024, 1, 1),
        initial_exchange_date=ActusDateTime(2024, 1, 15),
        maturity_date=ActusDateTime(2025, 1, 15),
        currency="USD",
        notional_principal=12_000.0,
        nominal_interest_rate=0.06,
        day_count_convention=DayCountConvention.A360,
        interest_payment_cycle="1M",
        principal_redemption_cycle="1M",
        next_principal_redemption_amount=1_000.0,
    )


def _simulate_python_path(attrs, rf_observer):
    """Run simulation through the standard Python path."""
    contract = LinearAmortizerContract(attrs, rf_observer)
    result = contract.simulate()
    return result


def _simulate_array_path(attrs, rf_observer):
    """Run simulation through the array-mode path."""
    arrays = precompute_lam_arrays(attrs, rf_observer)
    initial_state, event_types, year_fractions, rf_values, params = arrays
    final_state, payoffs = simulate_lam_array(
        initial_state, event_types, year_fractions, rf_values, params
    )
    return final_state, payoffs


def _assert_payoffs_match(py_result, payoffs, atol=ATOL):
    """Assert that Python and array payoffs match within tolerance."""
    py_payoffs = jnp.array([float(e.payoff) for e in py_result.events])
    assert py_payoffs.shape == payoffs.shape, (
        f"Event count mismatch: Python={py_payoffs.shape[0]}, Array={payoffs.shape[0]}"
    )
    for i in range(len(py_result.events)):
        assert abs(float(payoffs[i]) - float(py_payoffs[i])) <= atol, (
            f"Event {i} ({py_result.events[i].event_type.name}): "
            f"array={float(payoffs[i]):.2f}, python={float(py_payoffs[i]):.2f}"
        )


# ============================================================================
# End-to-end equivalence tests
# ============================================================================


class TestScanEquivalence:
    """End-to-end equivalence: simulate_lam_array vs contract.simulate()."""

    def test_fixed_rate_lam(self):
        """Fixed-rate 5-year LAM, annual payments, explicit PRNXT."""
        attrs = _make_fixed_lam_attrs()
        rf_obs = ConstantRiskFactorObserver(constant_value=0.0)

        py_result = _simulate_python_path(attrs, rf_obs)
        _, payoffs = _simulate_array_path(attrs, rf_obs)
        _assert_payoffs_match(py_result, payoffs)

    def test_fixed_rate_lam_rpl(self):
        """Fixed-rate LAM from borrower perspective (RPL)."""
        attrs = _make_fixed_lam_attrs(role=ContractRole.RPL)
        rf_obs = ConstantRiskFactorObserver(constant_value=0.0)

        py_result = _simulate_python_path(attrs, rf_obs)
        _, payoffs = _simulate_array_path(attrs, rf_obs)
        _assert_payoffs_match(py_result, payoffs)

    def test_auto_calculated_prnxt(self):
        """LAM with auto-calculated PRNXT (NT / n_periods)."""
        attrs = _make_auto_prnxt_attrs()
        rf_obs = ConstantRiskFactorObserver(constant_value=0.0)

        py_result = _simulate_python_path(attrs, rf_obs)
        _, payoffs = _simulate_array_path(attrs, rf_obs)
        _assert_payoffs_match(py_result, payoffs)

    def test_variable_rate_lam(self):
        """Variable-rate LAM with rate resets."""
        attrs = _make_variable_rate_lam_attrs()

        ts_data = {
            "LIBOR-6M": [
                (ActusDateTime(2024, 1, 1), 0.04),
                (ActusDateTime(2024, 7, 1), 0.045),
                (ActusDateTime(2025, 1, 1), 0.05),
                (ActusDateTime(2025, 7, 1), 0.048),
                (ActusDateTime(2026, 1, 1), 0.042),
                (ActusDateTime(2026, 7, 1), 0.04),
            ]
        }
        rf_obs = TimeSeriesRiskFactorObserver(risk_factors=ts_data)

        py_result = _simulate_python_path(attrs, rf_obs)
        _, payoffs = _simulate_array_path(attrs, rf_obs)
        _assert_payoffs_match(py_result, payoffs)

    def test_midlife_contract(self):
        """Mid-life LAM (IED < SD) — state reconstruction."""
        attrs = _make_midlife_lam_attrs()
        rf_obs = ConstantRiskFactorObserver(constant_value=0.0)

        py_result = _simulate_python_path(attrs, rf_obs)
        _, payoffs = _simulate_array_path(attrs, rf_obs)
        _assert_payoffs_match(py_result, payoffs)

    def test_monthly_payments(self):
        """LAM with monthly payment cycle."""
        attrs = _make_monthly_lam_attrs()
        rf_obs = ConstantRiskFactorObserver(constant_value=0.0)

        py_result = _simulate_python_path(attrs, rf_obs)
        _, payoffs = _simulate_array_path(attrs, rf_obs)
        _assert_payoffs_match(py_result, payoffs)

    def test_different_dcc_a365(self):
        """Test with A/365 day count convention."""
        attrs = _make_fixed_lam_attrs(dcc=DayCountConvention.A365)
        rf_obs = ConstantRiskFactorObserver(constant_value=0.0)

        py_result = _simulate_python_path(attrs, rf_obs)
        _, payoffs = _simulate_array_path(attrs, rf_obs)
        _assert_payoffs_match(py_result, payoffs)

    def test_different_dcc_30e360(self):
        """Test with 30E/360 day count convention."""
        attrs = _make_fixed_lam_attrs(dcc=DayCountConvention.E30360)
        rf_obs = ConstantRiskFactorObserver(constant_value=0.0)

        py_result = _simulate_python_path(attrs, rf_obs)
        _, payoffs = _simulate_array_path(attrs, rf_obs)
        _assert_payoffs_match(py_result, payoffs)

    def test_principal_reduces_each_period(self):
        """Verify notional reduces by PRNXT each period."""
        attrs = _make_fixed_lam_attrs(notional=100_000.0, prnxt=20_000.0, years=5)
        rf_obs = ConstantRiskFactorObserver(constant_value=0.0)

        final_state, payoffs = _simulate_array_path(attrs, rf_obs)
        # After 5 PR events of 20000 each, notional should be 0
        assert abs(float(final_state.nt)) <= ATOL

    def test_final_state_equivalence(self):
        """Final state fields should match between paths."""
        attrs = _make_fixed_lam_attrs()
        rf_obs = ConstantRiskFactorObserver(constant_value=0.0)

        py_result = _simulate_python_path(attrs, rf_obs)
        final_state, _ = _simulate_array_path(attrs, rf_obs)

        py_final = py_result.final_state
        assert abs(float(final_state.nt) - float(py_final.nt)) <= ATOL
        assert abs(float(final_state.ipnr) - float(py_final.ipnr)) <= 1e-6
        assert abs(float(final_state.ipac) - float(py_final.ipac)) <= ATOL

    def test_total_cashflow_equivalence(self):
        """Total cashflow should match between paths."""
        attrs = _make_fixed_lam_attrs()
        rf_obs = ConstantRiskFactorObserver(constant_value=0.0)

        py_result = _simulate_python_path(attrs, rf_obs)
        _, payoffs = _simulate_array_path(attrs, rf_obs)

        py_total = sum(float(e.payoff) for e in py_result.events)
        array_total = float(jnp.sum(payoffs))
        assert abs(array_total - py_total) <= ATOL

    def test_quarterly_payments(self):
        """LAM with quarterly IP and PR cycles."""
        attrs = ContractAttributes(
            contract_id="LAM-QUARTERLY",
            contract_type=ContractType.LAM,
            contract_role=ContractRole.RPA,
            status_date=ActusDateTime(2024, 1, 1),
            initial_exchange_date=ActusDateTime(2024, 1, 15),
            maturity_date=ActusDateTime(2026, 1, 15),
            currency="USD",
            notional_principal=80_000.0,
            nominal_interest_rate=0.04,
            day_count_convention=DayCountConvention.A360,
            interest_payment_cycle="3M",
            principal_redemption_cycle="3M",
            next_principal_redemption_amount=10_000.0,
        )
        rf_obs = ConstantRiskFactorObserver(constant_value=0.0)

        py_result = _simulate_python_path(attrs, rf_obs)
        _, payoffs = _simulate_array_path(attrs, rf_obs)
        _assert_payoffs_match(py_result, payoffs)

    def test_variable_rate_with_cap_floor(self):
        """Variable-rate LAM with rate floor and cap."""
        attrs = ContractAttributes(
            contract_id="LAM-CAPFLOOR",
            contract_type=ContractType.LAM,
            contract_role=ContractRole.RPA,
            status_date=ActusDateTime(2024, 1, 1),
            initial_exchange_date=ActusDateTime(2024, 1, 15),
            maturity_date=ActusDateTime(2027, 1, 15),
            currency="USD",
            notional_principal=100_000.0,
            nominal_interest_rate=0.05,
            day_count_convention=DayCountConvention.A360,
            interest_payment_cycle="6M",
            principal_redemption_cycle="6M",
            rate_reset_cycle="6M",
            rate_reset_spread=0.01,
            rate_reset_multiplier=1.0,
            rate_reset_market_object="LIBOR",
            rate_reset_anchor=ActusDateTime(2024, 1, 15),
            rate_reset_floor=0.02,
            rate_reset_cap=0.08,
        )
        ts_data = {
            "LIBOR": [
                (ActusDateTime(2024, 1, 1), 0.04),    # 0.04*1+0.01=0.05 -> ok
                (ActusDateTime(2024, 7, 1), 0.10),     # 0.10*1+0.01=0.11 -> capped at 0.08
                (ActusDateTime(2025, 1, 1), -0.02),    # -0.02*1+0.01=-0.01 -> floored at 0.02
                (ActusDateTime(2025, 7, 1), 0.06),     # 0.06*1+0.01=0.07 -> ok
                (ActusDateTime(2026, 1, 1), 0.05),
                (ActusDateTime(2026, 7, 1), 0.04),
            ]
        }
        rf_obs = TimeSeriesRiskFactorObserver(risk_factors=ts_data)

        py_result = _simulate_python_path(attrs, rf_obs)
        _, payoffs = _simulate_array_path(attrs, rf_obs)
        _assert_payoffs_match(py_result, payoffs)


# ============================================================================
# Batch tests
# ============================================================================


class TestBatchEquivalence:
    """Batch LAM simulation equivalence tests."""

    def test_batch_matches_individual(self):
        """Batch simulation should match individual simulations."""
        rf_obs = ConstantRiskFactorObserver(0.0)
        contracts = [
            (_make_fixed_lam_attrs(notional=100_000.0, prnxt=20_000.0), rf_obs),
            (_make_fixed_lam_attrs(notional=50_000.0, prnxt=10_000.0, role=ContractRole.RPL), rf_obs),
            (_make_monthly_lam_attrs(), rf_obs),
        ]

        # Individual simulations
        individual_totals = []
        for attrs, obs in contracts:
            _, payoffs = _simulate_array_path(attrs, obs)
            individual_totals.append(float(jnp.sum(payoffs)))

        # Batch simulation
        result = simulate_lam_portfolio(contracts)
        batch_totals = result["total_cashflows"]

        for i in range(len(contracts)):
            assert abs(float(batch_totals[i]) - individual_totals[i]) <= ATOL, (
                f"Contract {i}: batch={float(batch_totals[i]):.2f}, "
                f"individual={individual_totals[i]:.2f}"
            )

    def test_portfolio_count(self):
        """Portfolio should report correct number of contracts."""
        rf_obs = ConstantRiskFactorObserver(0.0)
        contracts = [
            (_make_fixed_lam_attrs(), rf_obs),
            (_make_monthly_lam_attrs(), rf_obs),
        ]
        result = simulate_lam_portfolio(contracts)
        assert result["num_contracts"] == 2


# ============================================================================
# JIT and vmap tests
# ============================================================================


class TestJITCompilation:
    """Test JIT compilation works correctly."""

    def test_jit_matches_eager(self):
        """JIT-compiled version should match eager execution."""
        attrs = _make_fixed_lam_attrs()
        rf_obs = ConstantRiskFactorObserver(0.0)

        arrays = precompute_lam_arrays(attrs, rf_obs)
        _, payoffs_eager = simulate_lam_array(*arrays)
        _, payoffs_jit = simulate_lam_array_jit(*arrays)

        assert jnp.allclose(payoffs_eager, payoffs_jit, atol=1e-6)


# ============================================================================
# Gradient tests
# ============================================================================


class TestGradients:
    """Test that gradients can be computed through LAM simulation."""

    def test_gradient_wrt_rate(self):
        """dTotalCashflow/dRate should be finite and non-zero."""
        attrs = _make_fixed_lam_attrs()
        rf_obs = ConstantRiskFactorObserver(0.0)

        arrays = precompute_lam_arrays(attrs, rf_obs)
        state, et, yf, rf, params = arrays

        def total_cashflow(rate):
            p = params._replace(nominal_interest_rate=rate)
            s = state._replace(ipnr=rate)
            _, payoffs = simulate_lam_array(s, et, yf, rf, p)
            return jnp.sum(payoffs)

        grad = jax.grad(total_cashflow)(params.nominal_interest_rate)
        assert jnp.isfinite(grad)
        assert float(grad) != 0.0
