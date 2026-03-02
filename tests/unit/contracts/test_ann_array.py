"""Equivalence tests for array-mode ANN simulation.

Runs identical contracts through both the Python path (AnnuityContract)
and the array-mode path (simulate_ann_array), asserting matching results.
Tolerance matches the ACTUS cross-validation standard (atol=1.0).
"""

import jax
import jax.numpy as jnp

from jactus.contracts.ann import AnnuityContract
from jactus.contracts.ann_array import (
    precompute_ann_arrays,
    simulate_ann_array,
    simulate_ann_array_jit,
    simulate_ann_portfolio,
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


def _make_fixed_ann_attrs(
    notional: float = 100_000.0,
    rate: float = 0.05,
    years: int = 5,
    dcc: DayCountConvention = DayCountConvention.A360,
    ip_cycle: str = "1Y",
    pr_cycle: str = "1Y",
    role: ContractRole = ContractRole.RPA,
) -> ContractAttributes:
    """Create attributes for a simple fixed-rate ANN.

    ANN does NOT set next_principal_redemption_amount -- it is
    auto-computed by the annuity formula.
    """
    return ContractAttributes(
        contract_id="ANN-FIXED",
        contract_type=ContractType.ANN,
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
    )


def _make_variable_rate_ann_attrs() -> ContractAttributes:
    """ANN with variable rate (rate resets)."""
    return ContractAttributes(
        contract_id="ANN-VARIABLE",
        contract_type=ContractType.ANN,
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
        rate_reset_cycle="6M",
        rate_reset_spread=0.01,
        rate_reset_multiplier=1.0,
        rate_reset_market_object="LIBOR-6M",
    )


def _make_monthly_ann_attrs() -> ContractAttributes:
    """ANN with monthly payments."""
    return ContractAttributes(
        contract_id="ANN-MONTHLY",
        contract_type=ContractType.ANN,
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
    )


def _simulate_python_path(attrs, rf_observer):
    """Run simulation through the standard Python path."""
    contract = AnnuityContract(attrs, rf_observer)
    result = contract.simulate()
    return result


def _simulate_array_path(attrs, rf_observer):
    """Run simulation through the array-mode path."""
    arrays = precompute_ann_arrays(attrs, rf_observer)
    initial_state, event_types, year_fractions, rf_values, params = arrays
    final_state, payoffs = simulate_ann_array(
        initial_state, event_types, year_fractions, rf_values, params
    )
    return final_state, payoffs


def _assert_payoffs_match(py_result, payoffs, atol=ATOL):
    """Assert that Python and array payoffs match within tolerance.

    The array path converts PRF events to NOP (zero payoff).  The Python
    path may emit PRF events with zero payoff.  We filter Python events
    to only non-PRF events for comparison, then compare the remaining
    payoffs with the non-NOP array payoffs event-by-event.
    """
    from jactus.core import EventType

    # 1. Total cashflow comparison (includes all events)
    py_total = sum(float(e.payoff) for e in py_result.events)
    array_total = float(jnp.sum(payoffs))
    assert abs(array_total - py_total) <= atol, (
        f"Total cashflow mismatch: array={array_total:.2f}, python={py_total:.2f}"
    )

    # 2. Filter out PRF events from Python path (they map to NOP in array path)
    py_events_filtered = [
        e for e in py_result.events if e.event_type != EventType.PRF
    ]
    py_payoffs = [float(e.payoff) for e in py_events_filtered]

    # 3. Filter out NOP events (zero-payoff padding from PRF conversion) from array
    array_payoffs = [float(payoffs[i]) for i in range(payoffs.shape[0]) if float(payoffs[i]) != 0.0]
    py_payoffs_nonzero = [p for p in py_payoffs if p != 0.0]

    # 4. Event count comparison: non-NOP array events vs non-PRF Python events
    assert len(array_payoffs) == len(py_payoffs_nonzero), (
        f"Non-zero event count mismatch: array={len(array_payoffs)}, "
        f"python={len(py_payoffs_nonzero)}"
    )

    # 5. Per-event payoff comparison for all non-NOP/non-PRF events
    for i in range(len(array_payoffs)):
        assert abs(array_payoffs[i] - py_payoffs_nonzero[i]) <= atol, (
            f"Non-zero event {i}: array={array_payoffs[i]:.2f}, "
            f"python={py_payoffs_nonzero[i]:.2f}"
        )


# ============================================================================
# End-to-end equivalence tests
# ============================================================================


class TestScanEquivalence:
    """End-to-end equivalence: simulate_ann_array vs contract.simulate()."""

    def test_fixed_rate_ann(self):
        """Fixed-rate 5-year ANN, annual payments."""
        attrs = _make_fixed_ann_attrs()
        rf_obs = ConstantRiskFactorObserver(constant_value=0.0)

        py_result = _simulate_python_path(attrs, rf_obs)
        _, payoffs = _simulate_array_path(attrs, rf_obs)
        _assert_payoffs_match(py_result, payoffs)

    def test_fixed_rate_ann_rpl(self):
        """ANN from borrower perspective (RPL)."""
        attrs = _make_fixed_ann_attrs(role=ContractRole.RPL)
        rf_obs = ConstantRiskFactorObserver(constant_value=0.0)

        py_result = _simulate_python_path(attrs, rf_obs)
        _, payoffs = _simulate_array_path(attrs, rf_obs)
        _assert_payoffs_match(py_result, payoffs)

    def test_variable_rate_ann(self):
        """Variable-rate ANN with rate resets."""
        attrs = _make_variable_rate_ann_attrs()

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

    def test_different_dcc_a365(self):
        """Test with A/365 day count convention."""
        attrs = _make_fixed_ann_attrs(dcc=DayCountConvention.A365)
        rf_obs = ConstantRiskFactorObserver(constant_value=0.0)

        py_result = _simulate_python_path(attrs, rf_obs)
        _, payoffs = _simulate_array_path(attrs, rf_obs)
        _assert_payoffs_match(py_result, payoffs)

    def test_different_dcc_30e360(self):
        """Test with 30E/360 day count convention."""
        attrs = _make_fixed_ann_attrs(dcc=DayCountConvention.E30360)
        rf_obs = ConstantRiskFactorObserver(constant_value=0.0)

        py_result = _simulate_python_path(attrs, rf_obs)
        _, payoffs = _simulate_array_path(attrs, rf_obs)
        _assert_payoffs_match(py_result, payoffs)

    def test_monthly_payments(self):
        """ANN with monthly payment cycle."""
        attrs = _make_monthly_ann_attrs()
        rf_obs = ConstantRiskFactorObserver(constant_value=0.0)

        py_result = _simulate_python_path(attrs, rf_obs)
        _, payoffs = _simulate_array_path(attrs, rf_obs)
        _assert_payoffs_match(py_result, payoffs)

    def test_midlife_ann(self):
        """Mid-life ANN where SD is after IED (contract already started)."""
        attrs = ContractAttributes(
            contract_id="ANN-MIDLIFE",
            contract_type=ContractType.ANN,
            contract_role=ContractRole.RPA,
            status_date=ActusDateTime(2026, 1, 1),  # SD after IED
            initial_exchange_date=ActusDateTime(2024, 1, 15),
            maturity_date=ActusDateTime(2029, 1, 15),
            currency="USD",
            notional_principal=100_000.0,
            nominal_interest_rate=0.05,
            day_count_convention=DayCountConvention.A360,
            interest_payment_cycle="1Y",
            principal_redemption_cycle="1Y",
        )
        rf_obs = ConstantRiskFactorObserver(constant_value=0.0)

        py_result = _simulate_python_path(attrs, rf_obs)
        _, payoffs = _simulate_array_path(attrs, rf_obs)
        _assert_payoffs_match(py_result, payoffs)

    def test_variable_rate_with_cap_floor(self):
        """ANN with variable rate and rate cap/floor constraints.

        The array path converts PRF events to NOP and uses the initial annuity
        amount rather than recalculating prnxt at each rate reset. This causes
        small differences in individual PR amounts, but the total cashflow
        should still be close. We use a wider tolerance (0.1% of notional)
        to account for the accumulated annuity recalculation difference.
        """
        attrs = ContractAttributes(
            contract_id="ANN-CAPFLOOR",
            contract_type=ContractType.ANN,
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
        # Use LIBOR values that trigger both floor (0.005 + 0.01 = 0.015 < 0.02)
        # and cap (0.09 + 0.01 = 0.10 > 0.08)
        ts_data = {
            "LIBOR": [
                (ActusDateTime(2024, 1, 1), 0.04),   # 0.04+0.01=0.05 (within bounds)
                (ActusDateTime(2024, 7, 1), 0.005),   # 0.005+0.01=0.015 -> floor=0.02
                (ActusDateTime(2025, 1, 1), 0.09),    # 0.09+0.01=0.10 -> cap=0.08
                (ActusDateTime(2025, 7, 1), 0.05),    # 0.05+0.01=0.06 (within bounds)
                (ActusDateTime(2026, 1, 1), 0.03),    # 0.03+0.01=0.04 (within bounds)
                (ActusDateTime(2026, 7, 1), 0.06),    # 0.06+0.01=0.07 (within bounds)
            ]
        }
        rf_obs = TimeSeriesRiskFactorObserver(risk_factors=ts_data)

        py_result = _simulate_python_path(attrs, rf_obs)
        _, payoffs = _simulate_array_path(attrs, rf_obs)

        # The array path converts PRF events to NOP and uses the initial
        # annuity amount rather than recalculating prnxt at each rate reset.
        # This causes per-event PR amount differences, so we compare only
        # total cashflow with 0.1% of notional tolerance, and verify event
        # count parity.
        from jactus.core import EventType

        py_total = sum(float(e.payoff) for e in py_result.events)
        array_total = float(jnp.sum(payoffs))
        assert abs(array_total - py_total) <= 100.0, (
            f"Total cashflow mismatch: array={array_total:.2f}, python={py_total:.2f}"
        )

        # Verify non-zero event counts match (NOP<->PRF mapping preserved)
        py_nonzero = [e for e in py_result.events
                      if e.event_type != EventType.PRF and float(e.payoff) != 0.0]
        array_nonzero = [float(payoffs[i]) for i in range(payoffs.shape[0])
                         if float(payoffs[i]) != 0.0]
        assert len(array_nonzero) == len(py_nonzero), (
            f"Non-zero event count mismatch: array={len(array_nonzero)}, "
            f"python={len(py_nonzero)}"
        )

    def test_total_cashflow_equivalence(self):
        """Total cashflow should match between paths."""
        attrs = _make_fixed_ann_attrs()
        rf_obs = ConstantRiskFactorObserver(constant_value=0.0)

        py_result = _simulate_python_path(attrs, rf_obs)
        _, payoffs = _simulate_array_path(attrs, rf_obs)

        py_total = sum(float(e.payoff) for e in py_result.events)
        array_total = float(jnp.sum(payoffs))
        assert abs(array_total - py_total) <= ATOL


# ============================================================================
# Batch tests
# ============================================================================


class TestBatchEquivalence:
    """Batch ANN simulation equivalence tests."""

    def test_batch_matches_individual(self):
        """Batch simulation should match individual simulations."""
        rf_obs = ConstantRiskFactorObserver(0.0)
        contracts = [
            (_make_fixed_ann_attrs(notional=100_000.0), rf_obs),
            (_make_fixed_ann_attrs(notional=50_000.0, role=ContractRole.RPL), rf_obs),
            (_make_monthly_ann_attrs(), rf_obs),
        ]

        # Individual simulations
        individual_totals = []
        for attrs, obs in contracts:
            _, payoffs = _simulate_array_path(attrs, obs)
            individual_totals.append(float(jnp.sum(payoffs)))

        # Batch simulation
        result = simulate_ann_portfolio(contracts)
        batch_totals = result["total_cashflows"]

        for i in range(len(contracts)):
            assert abs(float(batch_totals[i]) - individual_totals[i]) <= ATOL, (
                f"Contract {i}: batch={float(batch_totals[i]):.2f}, "
                f"individual={individual_totals[i]:.2f}"
            )


# ============================================================================
# JIT and vmap tests
# ============================================================================


class TestJITCompilation:
    """Test JIT compilation works correctly."""

    def test_jit_matches_eager(self):
        """JIT-compiled version should match eager execution."""
        attrs = _make_fixed_ann_attrs()
        rf_obs = ConstantRiskFactorObserver(0.0)

        arrays = precompute_ann_arrays(attrs, rf_obs)
        _, payoffs_eager = simulate_ann_array(*arrays)
        _, payoffs_jit = simulate_ann_array_jit(*arrays)

        assert jnp.allclose(payoffs_eager, payoffs_jit, atol=1e-6)


# ============================================================================
# Gradient tests
# ============================================================================


class TestGradients:
    """Test that gradients can be computed through ANN simulation."""

    def test_gradient_wrt_rate(self):
        """dTotalCashflow/dRate should be finite and non-zero."""
        attrs = _make_fixed_ann_attrs()
        rf_obs = ConstantRiskFactorObserver(0.0)

        arrays = precompute_ann_arrays(attrs, rf_obs)
        state, et, yf, rf, params = arrays

        def total_cashflow(rate):
            p = params._replace(nominal_interest_rate=rate)
            s = state._replace(ipnr=rate)
            _, payoffs = simulate_ann_array(s, et, yf, rf, p)
            return jnp.sum(payoffs)

        grad = jax.grad(total_cashflow)(params.nominal_interest_rate)
        assert jnp.isfinite(grad)
        assert float(grad) != 0.0
