"""Equivalence tests for array-mode SWPPV simulation.

Runs identical contracts through both the Python path (PlainVanillaSwapContract)
and the array-mode path (simulate_swppv_array), asserting matching results.
Tolerance matches the ACTUS cross-validation standard (atol=1.0).
"""

import jax
import jax.numpy as jnp
import pytest

from jactus.contracts.swppv import PlainVanillaSwapContract
from jactus.contracts.swppv_array import (
    SWPPVArrayParams,
    SWPPVArrayState,
    batch_simulate_swppv,
    precompute_swppv_arrays,
    prepare_swppv_batch,
    simulate_swppv_array,
    simulate_swppv_array_jit,
    simulate_swppv_portfolio,
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


def _make_fixed_vs_fixed_attrs(
    notional: float = 1_000_000.0,
    fixed_rate: float = 0.05,
    float_rate: float = 0.03,
    years: int = 5,
    dcc: DayCountConvention = DayCountConvention.A360,
    ip_cycle: str = "6M",
    role: ContractRole = ContractRole.RPA,
    ds: str = "S",
) -> ContractAttributes:
    """Create attributes for a fixed-vs-fixed SWPPV (no rate resets).

    With no rate_reset_cycle, the floating rate stays at nominal_interest_rate_2.
    Using delivery_settlement='S' for net IP events.
    """
    return ContractAttributes(
        contract_id="SWAP-FIXED",
        contract_type=ContractType.SWPPV,
        contract_role=role,
        status_date=ActusDateTime(2024, 1, 1),
        initial_exchange_date=ActusDateTime(2024, 1, 15),
        maturity_date=ActusDateTime(2024 + years, 1, 15),
        currency="USD",
        notional_principal=notional,
        nominal_interest_rate=fixed_rate,
        nominal_interest_rate_2=float_rate,
        day_count_convention=dcc,
        interest_payment_cycle=ip_cycle,
        delivery_settlement=ds,
    )


def _make_fixed_vs_float_attrs(
    notional: float = 1_000_000.0,
    fixed_rate: float = 0.05,
    initial_float_rate: float = 0.03,
    years: int = 3,
    dcc: DayCountConvention = DayCountConvention.A360,
    ip_cycle: str = "6M",
    rr_cycle: str = "3M",
    role: ContractRole = ContractRole.RPA,
    ds: str = "S",
) -> ContractAttributes:
    """Create attributes for a fixed-vs-floating SWPPV with rate resets."""
    return ContractAttributes(
        contract_id="SWAP-FLOAT",
        contract_type=ContractType.SWPPV,
        contract_role=role,
        status_date=ActusDateTime(2024, 1, 1),
        initial_exchange_date=ActusDateTime(2024, 1, 15),
        maturity_date=ActusDateTime(2024 + years, 1, 15),
        currency="USD",
        notional_principal=notional,
        nominal_interest_rate=fixed_rate,
        nominal_interest_rate_2=initial_float_rate,
        day_count_convention=dcc,
        interest_payment_cycle=ip_cycle,
        rate_reset_cycle=rr_cycle,
        rate_reset_anchor=ActusDateTime(2024, 4, 15),
        rate_reset_spread=0.01,
        rate_reset_multiplier=1.0,
        rate_reset_market_object="LIBOR-3M",
        delivery_settlement=ds,
    )


def _make_separate_settlement_attrs() -> ContractAttributes:
    """Create SWPPV with separate IPFX/IPFL events (delivery_settlement='D')."""
    return ContractAttributes(
        contract_id="SWAP-SEP",
        contract_type=ContractType.SWPPV,
        contract_role=ContractRole.RPA,
        status_date=ActusDateTime(2024, 1, 1),
        initial_exchange_date=ActusDateTime(2024, 1, 15),
        maturity_date=ActusDateTime(2027, 1, 15),
        currency="USD",
        notional_principal=1_000_000.0,
        nominal_interest_rate=0.05,
        nominal_interest_rate_2=0.03,
        day_count_convention=DayCountConvention.A360,
        interest_payment_cycle="6M",
        delivery_settlement="D",
    )


def _simulate_python_path(attrs, rf_observer):
    """Run simulation through the standard Python path."""
    contract = PlainVanillaSwapContract(attrs, rf_observer)
    result = contract.simulate()
    return result


def _simulate_array_path(attrs, rf_observer):
    """Run simulation through the array-mode path."""
    arrays = precompute_swppv_arrays(attrs, rf_observer)
    initial_state, event_types, year_fractions, rf_values, params = arrays
    final_state, payoffs = simulate_swppv_array(
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
    """End-to-end equivalence: simulate_swppv_array vs contract.simulate()."""

    def test_fixed_vs_fixed(self):
        """Fixed-vs-fixed swap with net settlement — no rate resets."""
        attrs = _make_fixed_vs_fixed_attrs()
        rf_obs = ConstantRiskFactorObserver(constant_value=0.0)

        py_result = _simulate_python_path(attrs, rf_obs)
        _, payoffs = _simulate_array_path(attrs, rf_obs)
        _assert_payoffs_match(py_result, payoffs)

    def test_fixed_vs_float(self):
        """Fixed-vs-floating swap with rate resets and net settlement."""
        attrs = _make_fixed_vs_float_attrs()

        ts_data = {
            "LIBOR-3M": [
                (ActusDateTime(2024, 1, 1), 0.03),
                (ActusDateTime(2024, 4, 1), 0.035),
                (ActusDateTime(2024, 7, 1), 0.04),
                (ActusDateTime(2025, 1, 1), 0.045),
                (ActusDateTime(2025, 7, 1), 0.04),
                (ActusDateTime(2026, 1, 1), 0.038),
                (ActusDateTime(2026, 7, 1), 0.035),
            ]
        }
        rf_obs = TimeSeriesRiskFactorObserver(risk_factors=ts_data)

        py_result = _simulate_python_path(attrs, rf_obs)
        _, payoffs = _simulate_array_path(attrs, rf_obs)
        _assert_payoffs_match(py_result, payoffs)

    def test_rpl_role(self):
        """Swap from RPL (pay fixed, receive floating) perspective."""
        attrs = _make_fixed_vs_fixed_attrs(role=ContractRole.RPL)
        rf_obs = ConstantRiskFactorObserver(constant_value=0.0)

        py_result = _simulate_python_path(attrs, rf_obs)
        _, payoffs = _simulate_array_path(attrs, rf_obs)
        _assert_payoffs_match(py_result, payoffs)

    def test_rfl_role(self):
        """Swap from RFL (receive floating leg) perspective."""
        attrs = _make_fixed_vs_fixed_attrs(role=ContractRole.RFL)
        rf_obs = ConstantRiskFactorObserver(constant_value=0.0)

        py_result = _simulate_python_path(attrs, rf_obs)
        _, payoffs = _simulate_array_path(attrs, rf_obs)
        _assert_payoffs_match(py_result, payoffs)

    def test_separate_settlement(self):
        """Swap with separate IPFX/IPFL settlement (delivery_settlement='D')."""
        attrs = _make_separate_settlement_attrs()
        rf_obs = ConstantRiskFactorObserver(constant_value=0.0)

        py_result = _simulate_python_path(attrs, rf_obs)
        _, payoffs = _simulate_array_path(attrs, rf_obs)
        _assert_payoffs_match(py_result, payoffs)

    def test_total_cashflow(self):
        """Total cashflow should match between paths."""
        attrs = _make_fixed_vs_fixed_attrs()
        rf_obs = ConstantRiskFactorObserver(constant_value=0.0)

        py_result = _simulate_python_path(attrs, rf_obs)
        _, payoffs = _simulate_array_path(attrs, rf_obs)

        py_total = sum(float(e.payoff) for e in py_result.events)
        array_total = float(jnp.sum(payoffs))
        assert abs(array_total - py_total) <= ATOL

    def test_final_state_equivalence(self):
        """Final state fields should match between paths."""
        attrs = _make_fixed_vs_fixed_attrs()
        rf_obs = ConstantRiskFactorObserver(constant_value=0.0)

        py_result = _simulate_python_path(attrs, rf_obs)
        final_state, _ = _simulate_array_path(attrs, rf_obs)

        py_final = py_result.final_state
        assert abs(float(final_state.nt) - float(py_final.nt)) <= ATOL
        assert abs(float(final_state.ipnr) - float(py_final.ipnr)) <= 1e-6

    def test_different_dcc_a365(self):
        """Test with A/365 day count convention."""
        attrs = _make_fixed_vs_fixed_attrs(dcc=DayCountConvention.A365)
        rf_obs = ConstantRiskFactorObserver(constant_value=0.0)

        py_result = _simulate_python_path(attrs, rf_obs)
        _, payoffs = _simulate_array_path(attrs, rf_obs)
        _assert_payoffs_match(py_result, payoffs)

    def test_different_dcc_30e360(self):
        """Test with 30E/360 day count convention."""
        attrs = _make_fixed_vs_fixed_attrs(dcc=DayCountConvention.E30360)
        rf_obs = ConstantRiskFactorObserver(constant_value=0.0)

        py_result = _simulate_python_path(attrs, rf_obs)
        _, payoffs = _simulate_array_path(attrs, rf_obs)
        _assert_payoffs_match(py_result, payoffs)

    def test_float_vs_float_rate_resets(self):
        """Floating-vs-floating swap where RR events change the floating rate."""
        attrs = _make_fixed_vs_float_attrs(
            fixed_rate=0.04,
            initial_float_rate=0.035,
            rr_cycle="6M",
        )

        ts_data = {
            "LIBOR-3M": [
                (ActusDateTime(2024, 1, 1), 0.035),
                (ActusDateTime(2024, 7, 1), 0.04),
                (ActusDateTime(2025, 1, 1), 0.042),
                (ActusDateTime(2025, 7, 1), 0.038),
                (ActusDateTime(2026, 1, 1), 0.035),
            ]
        }
        rf_obs = TimeSeriesRiskFactorObserver(risk_factors=ts_data)

        py_result = _simulate_python_path(attrs, rf_obs)
        _, payoffs = _simulate_array_path(attrs, rf_obs)
        _assert_payoffs_match(py_result, payoffs)

    def test_midlife_swap(self):
        """Mid-life SWPPV where SD is after IED (contract already started).

        The Python path returns all events from IED, while the array path
        only simulates post-SD events (with a partial first IP for the
        accrued period from SD to the next IP date). We compare by checking
        that:
        1. The array path produces the correct number of post-SD events
        2. The full IP events (after the first partial) match the Python path
        3. Final state is consistent
        """
        sd = ActusDateTime(2025, 7, 1)
        attrs = ContractAttributes(
            contract_id="SWPPV-MIDLIFE",
            contract_type=ContractType.SWPPV,
            contract_role=ContractRole.RFL,
            status_date=sd,  # SD after IED
            initial_exchange_date=ActusDateTime(2024, 1, 15),
            maturity_date=ActusDateTime(2027, 1, 15),
            currency="USD",
            notional_principal=1_000_000.0,
            nominal_interest_rate=0.03,
            nominal_interest_rate_2=0.04,
            day_count_convention=DayCountConvention.A360,
            interest_payment_cycle="6M",
            delivery_settlement="S",
        )
        rf_obs = ConstantRiskFactorObserver(constant_value=0.0)

        py_result = _simulate_python_path(attrs, rf_obs)
        final_state, payoffs = _simulate_array_path(attrs, rf_obs)

        # Array path should produce events (not empty)
        assert payoffs.shape[0] > 0, "Array path should produce events"

        # The array path has: [partial_IP, full_IP_1, full_IP_2, ..., MD]
        # The Python post-SD has: [full_IP_0, full_IP_1, full_IP_2, ..., MD]
        # where full_IP_0 spans before+after SD (so it's larger than the
        # array's partial_IP). The subsequent full IPs should match.
        py_post_sd_ips = [
            float(e.payoff) for e in py_result.events
            if e.event_time > sd and float(e.payoff) != 0.0
        ]
        array_full_ips = [
            float(payoffs[i]) for i in range(1, payoffs.shape[0])
            if float(payoffs[i]) != 0.0
        ]

        # Array full IPs should match Python post-SD IPs[1:] (both skip
        # the SD-boundary IP)
        assert len(array_full_ips) == len(py_post_sd_ips) - 1, (
            f"Full IP count mismatch: array={len(array_full_ips)}, "
            f"python (excl first)={len(py_post_sd_ips) - 1}"
        )
        for i in range(len(array_full_ips)):
            assert abs(array_full_ips[i] - py_post_sd_ips[i + 1]) <= ATOL, (
                f"Full IP event {i}: array={array_full_ips[i]:.2f}, "
                f"python={py_post_sd_ips[i + 1]:.2f}"
            )

        # Final state: notional should be zero (swap expired at MD)
        assert abs(float(final_state.nt)) <= ATOL

    def test_rate_cap_floor(self):
        """SWPPV with rate cap/floor on floating leg via rate resets."""
        attrs = ContractAttributes(
            contract_id="SWAP-CAPFLOOR",
            contract_type=ContractType.SWPPV,
            contract_role=ContractRole.RPA,
            status_date=ActusDateTime(2024, 1, 1),
            initial_exchange_date=ActusDateTime(2024, 1, 15),
            maturity_date=ActusDateTime(2027, 1, 15),
            currency="USD",
            notional_principal=1_000_000.0,
            nominal_interest_rate=0.05,
            nominal_interest_rate_2=0.03,
            day_count_convention=DayCountConvention.A360,
            interest_payment_cycle="6M",
            rate_reset_cycle="6M",
            rate_reset_anchor=ActusDateTime(2024, 1, 15),
            rate_reset_spread=0.01,
            rate_reset_multiplier=1.0,
            rate_reset_market_object="LIBOR-6M",
            rate_reset_floor=0.02,
            rate_reset_cap=0.08,
            delivery_settlement="S",
        )
        # Use LIBOR values that trigger both floor and cap
        ts_data = {
            "LIBOR-6M": [
                (ActusDateTime(2024, 1, 1), 0.03),    # 0.03+0.01=0.04 (within bounds)
                (ActusDateTime(2024, 7, 1), 0.005),   # 0.005+0.01=0.015 -> floor=0.02
                (ActusDateTime(2025, 1, 1), 0.09),    # 0.09+0.01=0.10 -> cap=0.08
                (ActusDateTime(2025, 7, 1), 0.05),    # 0.05+0.01=0.06 (within bounds)
                (ActusDateTime(2026, 1, 1), 0.04),    # 0.04+0.01=0.05 (within bounds)
                (ActusDateTime(2026, 7, 1), 0.06),    # 0.06+0.01=0.07 (within bounds)
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
    """Batch SWPPV simulation equivalence tests."""

    def test_batch_matches_individual(self):
        """Batch simulation should match individual simulations."""
        rf_obs = ConstantRiskFactorObserver(0.0)
        contracts = [
            (_make_fixed_vs_fixed_attrs(notional=1_000_000.0), rf_obs),
            (_make_fixed_vs_fixed_attrs(notional=500_000.0, role=ContractRole.RPL), rf_obs),
            (_make_separate_settlement_attrs(), rf_obs),
        ]

        # Individual simulations
        individual_totals = []
        for attrs, obs in contracts:
            _, payoffs = _simulate_array_path(attrs, obs)
            individual_totals.append(float(jnp.sum(payoffs)))

        # Batch simulation
        result = simulate_swppv_portfolio(contracts)
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
        attrs = _make_fixed_vs_fixed_attrs()
        rf_obs = ConstantRiskFactorObserver(0.0)

        arrays = precompute_swppv_arrays(attrs, rf_obs)
        _, payoffs_eager = simulate_swppv_array(*arrays)
        _, payoffs_jit = simulate_swppv_array_jit(*arrays)

        assert jnp.allclose(payoffs_eager, payoffs_jit, atol=1e-6)


# ============================================================================
# Gradient tests
# ============================================================================


class TestGradients:
    """Test that gradients can be computed through SWPPV simulation."""

    def test_gradient_wrt_fixed_rate(self):
        """dTotalCashflow/dFixedRate should be finite and non-zero."""
        attrs = _make_fixed_vs_fixed_attrs()
        rf_obs = ConstantRiskFactorObserver(0.0)

        arrays = precompute_swppv_arrays(attrs, rf_obs)
        state, et, yf, rf, params = arrays

        def total_cashflow(rate):
            p = params._replace(fixed_rate=rate)
            _, payoffs = simulate_swppv_array(state, et, yf, rf, p)
            return jnp.sum(payoffs)

        grad = jax.grad(total_cashflow)(params.fixed_rate)
        assert jnp.isfinite(grad)
        assert float(grad) != 0.0
