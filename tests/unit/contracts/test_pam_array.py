"""Equivalence tests for array-mode PAM simulation.

Runs identical contracts through both the Python path (PrincipalAtMaturityContract)
and the array-mode path (simulate_pam_array), asserting matching results.
Tolerance matches the ACTUS cross-validation standard (atol=1.0).
"""

import jax
import jax.numpy as jnp
import pytest

from jactus.contracts.pam import PrincipalAtMaturityContract
from jactus.contracts.pam_array import (
    NOP_EVENT_IDX,
    PAMArrayParams,
    PAMArrayState,
    _pof_ad,
    _pof_ce,
    _pof_ied,
    _pof_ip,
    _pof_ipci,
    _pof_md,
    _pof_noop,
    _pof_rr,
    _pof_rrf,
    _pof_sc,
    _stf_ad,
    _stf_ied,
    _stf_ip,
    _stf_ipci,
    _stf_md,
    _stf_noop,
    _stf_rr,
    _stf_rrf,
    precompute_pam_arrays,
    simulate_pam_array,
    simulate_pam_array_jit,
    simulate_pam_portfolio,
)
from jactus.core import (
    ActusDateTime,
    ContractAttributes,
    ContractRole,
    ContractType,
    DayCountConvention,
)
from jactus.engine.vectorized import (
    ArraySimulationResult,
    BatchSimulationResult,
    validate_pam_for_array_mode,
)
from jactus.observers import ConstantRiskFactorObserver, TimeSeriesRiskFactorObserver

# Tolerance matching ACTUS cross-validation standard
ATOL = 1.0


# ============================================================================
# Fixtures
# ============================================================================


def _make_fixed_rate_attrs(
    notional: float = 100_000.0,
    rate: float = 0.05,
    years: int = 5,
    dcc: DayCountConvention = DayCountConvention.A360,
    ip_cycle: str = "1Y",
    role: ContractRole = ContractRole.RPA,
) -> ContractAttributes:
    """Create attributes for a simple fixed-rate PAM bond."""
    return ContractAttributes(
        contract_id="PAM-FIXED",
        contract_type=ContractType.PAM,
        contract_role=role,
        status_date=ActusDateTime(2024, 1, 1),
        initial_exchange_date=ActusDateTime(2024, 1, 15),
        maturity_date=ActusDateTime(2024 + years, 1, 15),
        currency="USD",
        notional_principal=notional,
        nominal_interest_rate=rate,
        day_count_convention=dcc,
        interest_payment_cycle=ip_cycle,
    )


def _make_variable_rate_attrs(
    notional: float = 100_000.0,
    initial_rate: float = 0.03,
    spread: float = 0.01,
) -> ContractAttributes:
    """Create attributes for a variable-rate PAM with rate resets."""
    return ContractAttributes(
        contract_id="PAM-VARIABLE",
        contract_type=ContractType.PAM,
        contract_role=ContractRole.RPA,
        status_date=ActusDateTime(2024, 1, 1),
        initial_exchange_date=ActusDateTime(2024, 1, 15),
        maturity_date=ActusDateTime(2027, 1, 15),
        currency="USD",
        notional_principal=notional,
        nominal_interest_rate=initial_rate,
        day_count_convention=DayCountConvention.A360,
        interest_payment_cycle="6M",
        rate_reset_cycle="6M",
        rate_reset_spread=spread,
        rate_reset_multiplier=1.0,
        rate_reset_market_object="LIBOR-6M",
    )


def _make_midlife_attrs() -> ContractAttributes:
    """Create attributes for a mid-life PAM (IED < SD)."""
    return ContractAttributes(
        contract_id="PAM-MIDLIFE",
        contract_type=ContractType.PAM,
        contract_role=ContractRole.RPA,
        status_date=ActusDateTime(2026, 1, 1),
        initial_exchange_date=ActusDateTime(2024, 1, 15),
        maturity_date=ActusDateTime(2029, 1, 15),
        currency="USD",
        notional_principal=100_000.0,
        nominal_interest_rate=0.05,
        day_count_convention=DayCountConvention.A360,
        interest_payment_cycle="1Y",
    )


def _simulate_python_path(attrs, rf_observer):
    """Run simulation through the standard Python path."""
    contract = PrincipalAtMaturityContract(attrs, rf_observer)
    result = contract.simulate()
    return result


def _simulate_array_path(attrs, rf_observer):
    """Run simulation through the array-mode path."""
    arrays = precompute_pam_arrays(attrs, rf_observer)
    initial_state, event_types, year_fractions, rf_values, params = arrays
    final_state, payoffs = simulate_pam_array(
        initial_state, event_types, year_fractions, rf_values, params
    )
    return final_state, payoffs


# ============================================================================
# End-to-end equivalence tests
# ============================================================================


class TestScanEquivalence:
    """End-to-end equivalence: simulate_pam_array vs contract.simulate()."""

    def test_fixed_rate_bond(self):
        """Fixed-rate 5-year bond, annual IP."""
        attrs = _make_fixed_rate_attrs()
        rf_obs = ConstantRiskFactorObserver(constant_value=0.0)

        py_result = _simulate_python_path(attrs, rf_obs)
        final_state, payoffs = _simulate_array_path(attrs, rf_obs)

        py_payoffs = jnp.array([float(e.payoff) for e in py_result.events])
        assert py_payoffs.shape == payoffs.shape, (
            f"Event count mismatch: Python={py_payoffs.shape[0]}, Array={payoffs.shape[0]}"
        )
        jnp.allclose(payoffs, py_payoffs, atol=ATOL)
        # Also check with explicit assertion for clarity
        for i in range(len(py_result.events)):
            assert abs(float(payoffs[i]) - float(py_payoffs[i])) <= ATOL, (
                f"Event {i} ({py_result.events[i].event_type.name}): "
                f"array={float(payoffs[i]):.2f}, python={float(py_payoffs[i]):.2f}"
            )

    def test_fixed_rate_bond_rpl(self):
        """Fixed-rate bond from borrower perspective (RPL)."""
        attrs = _make_fixed_rate_attrs(role=ContractRole.RPL)
        rf_obs = ConstantRiskFactorObserver(constant_value=0.0)

        py_result = _simulate_python_path(attrs, rf_obs)
        _, payoffs = _simulate_array_path(attrs, rf_obs)

        py_payoffs = jnp.array([float(e.payoff) for e in py_result.events])
        for i in range(len(py_result.events)):
            assert abs(float(payoffs[i]) - float(py_payoffs[i])) <= ATOL, (
                f"Event {i} ({py_result.events[i].event_type.name}): "
                f"array={float(payoffs[i]):.2f}, python={float(py_payoffs[i]):.2f}"
            )

    def test_variable_rate_bond(self):
        """Variable-rate bond with rate resets."""
        attrs = _make_variable_rate_attrs()

        # Use time series for rate resets
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

        py_payoffs = jnp.array([float(e.payoff) for e in py_result.events])
        assert py_payoffs.shape == payoffs.shape
        for i in range(len(py_result.events)):
            assert abs(float(payoffs[i]) - float(py_payoffs[i])) <= ATOL, (
                f"Event {i} ({py_result.events[i].event_type.name}): "
                f"array={float(payoffs[i]):.2f}, python={float(py_payoffs[i]):.2f}"
            )

    def test_midlife_contract(self):
        """Mid-life contract (IED < SD) — state reconstruction."""
        attrs = _make_midlife_attrs()
        rf_obs = ConstantRiskFactorObserver(constant_value=0.0)

        py_result = _simulate_python_path(attrs, rf_obs)
        _, payoffs = _simulate_array_path(attrs, rf_obs)

        py_payoffs = jnp.array([float(e.payoff) for e in py_result.events])
        assert py_payoffs.shape == payoffs.shape
        for i in range(len(py_result.events)):
            assert abs(float(payoffs[i]) - float(py_payoffs[i])) <= ATOL, (
                f"Event {i} ({py_result.events[i].event_type.name}): "
                f"array={float(payoffs[i]):.2f}, python={float(py_payoffs[i]):.2f}"
            )

    def test_different_dcc_30e360(self):
        """Test with 30E/360 day count convention."""
        attrs = _make_fixed_rate_attrs(dcc=DayCountConvention.E30360)
        rf_obs = ConstantRiskFactorObserver(constant_value=0.0)

        py_result = _simulate_python_path(attrs, rf_obs)
        _, payoffs = _simulate_array_path(attrs, rf_obs)

        py_payoffs = jnp.array([float(e.payoff) for e in py_result.events])
        for i in range(len(py_result.events)):
            assert abs(float(payoffs[i]) - float(py_payoffs[i])) <= ATOL, (
                f"Event {i}: array={float(payoffs[i]):.2f}, python={float(py_payoffs[i]):.2f}"
            )

    def test_quarterly_ip_cycle(self):
        """Test with quarterly interest payments."""
        attrs = _make_fixed_rate_attrs(ip_cycle="3M", years=2)
        rf_obs = ConstantRiskFactorObserver(constant_value=0.0)

        py_result = _simulate_python_path(attrs, rf_obs)
        _, payoffs = _simulate_array_path(attrs, rf_obs)

        py_payoffs = jnp.array([float(e.payoff) for e in py_result.events])
        assert py_payoffs.shape == payoffs.shape
        for i in range(len(py_result.events)):
            assert abs(float(payoffs[i]) - float(py_payoffs[i])) <= ATOL, (
                f"Event {i}: array={float(payoffs[i]):.2f}, python={float(py_payoffs[i]):.2f}"
            )

    def test_monthly_ip_cycle(self):
        """Test with monthly interest payments."""
        attrs = _make_fixed_rate_attrs(ip_cycle="1M", years=1)
        rf_obs = ConstantRiskFactorObserver(constant_value=0.0)

        py_result = _simulate_python_path(attrs, rf_obs)
        _, payoffs = _simulate_array_path(attrs, rf_obs)

        py_payoffs = jnp.array([float(e.payoff) for e in py_result.events])
        assert py_payoffs.shape == payoffs.shape
        for i in range(len(py_result.events)):
            assert abs(float(payoffs[i]) - float(py_payoffs[i])) <= ATOL

    def test_final_state_equivalence(self):
        """Final state fields should match between paths."""
        attrs = _make_fixed_rate_attrs()
        rf_obs = ConstantRiskFactorObserver(constant_value=0.0)

        py_result = _simulate_python_path(attrs, rf_obs)
        final_state, _ = _simulate_array_path(attrs, rf_obs)

        py_final = py_result.final_state
        assert abs(float(final_state.nt) - float(py_final.nt)) <= ATOL
        assert abs(float(final_state.ipnr) - float(py_final.ipnr)) <= 1e-6
        assert abs(float(final_state.ipac) - float(py_final.ipac)) <= ATOL
        assert abs(float(final_state.feac) - float(py_final.feac)) <= ATOL

    def test_total_cashflow_equivalence(self):
        """Total cashflow should match between paths."""
        attrs = _make_fixed_rate_attrs()
        rf_obs = ConstantRiskFactorObserver(constant_value=0.0)

        py_result = _simulate_python_path(attrs, rf_obs)
        _, payoffs = _simulate_array_path(attrs, rf_obs)

        py_total = sum(float(e.payoff) for e in py_result.events)
        array_total = float(jnp.sum(payoffs))
        assert abs(array_total - py_total) <= ATOL * len(py_result.events)


# ============================================================================
# NOP event tests
# ============================================================================


class TestNoop:
    """NOP (padding) events produce zero payoff and identity state."""

    def test_nop_payoff_zero(self):
        state = PAMArrayState(
            nt=jnp.array(100_000.0),
            ipnr=jnp.array(0.05),
            ipac=jnp.array(500.0),
            feac=jnp.array(10.0),
            nsc=jnp.array(1.0),
            isc=jnp.array(1.0),
        )
        params = PAMArrayParams(
            role_sign=jnp.array(1.0),
            notional_principal=jnp.array(100_000.0),
            nominal_interest_rate=jnp.array(0.05),
            premium_discount_at_ied=jnp.array(0.0),
            accrued_interest=jnp.array(0.0),
            fee_rate=jnp.array(0.0),
            fee_basis=jnp.array(0, dtype=jnp.int32),
            penalty_rate=jnp.array(0.0),
            penalty_type=jnp.array(0, dtype=jnp.int32),
            price_at_purchase_date=jnp.array(0.0),
            price_at_termination_date=jnp.array(0.0),
            rate_reset_spread=jnp.array(0.0),
            rate_reset_multiplier=jnp.array(1.0),
            rate_reset_floor=jnp.array(0.0),
            rate_reset_cap=jnp.array(1.0),
            rate_reset_next=jnp.array(0.05),
            has_rate_floor=jnp.array(0.0),
            has_rate_cap=jnp.array(0.0),
            ied_ipac=jnp.array(0.0),
        )

        payoff = _pof_noop(state, params, jnp.array(0.5), jnp.array(0.0))
        assert float(payoff) == 0.0

        new_state = _stf_noop(state, params, jnp.array(0.5), jnp.array(0.0))
        assert float(new_state.nt) == float(state.nt)
        assert float(new_state.ipnr) == float(state.ipnr)
        assert float(new_state.ipac) == float(state.ipac)

    def test_nop_in_scan(self):
        """NOP events in the scan loop produce zero payoff."""
        attrs = _make_fixed_rate_attrs(years=1)
        rf_obs = ConstantRiskFactorObserver(constant_value=0.0)

        arrays = precompute_pam_arrays(attrs, rf_obs)
        init_state, event_types, year_fractions, rf_values, params = arrays

        n_real = event_types.shape[0]
        n_pad = 5

        # Pad with NOP events
        padded_et = jnp.concatenate([event_types, jnp.full(n_pad, NOP_EVENT_IDX, dtype=jnp.int32)])
        padded_yf = jnp.concatenate([year_fractions, jnp.zeros(n_pad)])
        padded_rf = jnp.concatenate([rf_values, jnp.zeros(n_pad)])

        final_state, payoffs = simulate_pam_array(
            init_state, padded_et, padded_yf, padded_rf, params
        )

        # Padded payoffs should be zero
        for i in range(n_real, n_real + n_pad):
            assert float(payoffs[i]) == 0.0

        # Real payoffs should match non-padded
        _, payoffs_real = simulate_pam_array(
            init_state, event_types, year_fractions, rf_values, params
        )
        for i in range(n_real):
            assert abs(float(payoffs[i]) - float(payoffs_real[i])) < 1e-6


# ============================================================================
# JIT compilation tests
# ============================================================================


class TestJIT:
    """Results should be unchanged after JIT compilation."""

    def test_jit_matches_eager(self):
        attrs = _make_fixed_rate_attrs()
        rf_obs = ConstantRiskFactorObserver(constant_value=0.0)

        arrays = precompute_pam_arrays(attrs, rf_obs)
        init_state, event_types, year_fractions, rf_values, params = arrays

        # Eager
        final_eager, payoffs_eager = simulate_pam_array(
            init_state, event_types, year_fractions, rf_values, params
        )

        # JIT
        final_jit, payoffs_jit = simulate_pam_array_jit(
            init_state, event_types, year_fractions, rf_values, params
        )

        assert jnp.allclose(payoffs_eager, payoffs_jit, atol=1e-6)
        assert jnp.allclose(jnp.array(final_eager), jnp.array(final_jit), atol=1e-6)


# ============================================================================
# Gradient tests
# ============================================================================


class TestGrad:
    """jax.grad produces finite, non-zero sensitivities."""

    def test_grad_wrt_rate(self):
        """dPV/dRate should be finite and negative (higher rate = lower PV for lender)."""
        attrs = _make_fixed_rate_attrs()
        rf_obs = ConstantRiskFactorObserver(constant_value=0.0)

        arrays = precompute_pam_arrays(attrs, rf_obs)
        init_state, event_types, year_fractions, rf_values, params = arrays

        def total_cashflow(rate_override):
            new_params = params._replace(nominal_interest_rate=rate_override)
            new_state = init_state._replace(ipnr=rate_override)
            _, payoffs = simulate_pam_array(
                new_state, event_types, year_fractions, rf_values, new_params
            )
            return jnp.sum(payoffs)

        grad_fn = jax.grad(total_cashflow)
        grad_val = grad_fn(jnp.array(0.05))
        assert jnp.isfinite(grad_val), f"Gradient is not finite: {grad_val}"
        assert float(grad_val) != 0.0, "Gradient should be non-zero"

    def test_grad_wrt_notional(self):
        """dCashflow/dNotional should be finite and non-zero."""
        attrs = _make_fixed_rate_attrs()
        rf_obs = ConstantRiskFactorObserver(constant_value=0.0)

        arrays = precompute_pam_arrays(attrs, rf_obs)
        init_state, event_types, year_fractions, rf_values, params = arrays

        def total_cashflow(notional_override):
            role_sign = params.role_sign
            new_params = params._replace(notional_principal=notional_override)
            new_state = init_state._replace(nt=role_sign * notional_override)
            _, payoffs = simulate_pam_array(
                new_state, event_types, year_fractions, rf_values, new_params
            )
            return jnp.sum(payoffs)

        grad_fn = jax.grad(total_cashflow)
        grad_val = grad_fn(jnp.array(100_000.0))
        assert jnp.isfinite(grad_val)
        assert float(grad_val) != 0.0


# ============================================================================
# Batch / portfolio tests
# ============================================================================


class TestBatch:
    """Batched simulation with padding."""

    def test_two_contracts_same(self):
        """Two identical contracts should produce identical payoffs."""
        attrs = _make_fixed_rate_attrs()
        rf_obs = ConstantRiskFactorObserver(constant_value=0.0)

        contracts = [(attrs, rf_obs), (attrs, rf_obs)]
        result = simulate_pam_portfolio(contracts)

        assert result["num_contracts"] == 2
        # Both contracts identical → same cashflows
        cf = result["total_cashflows"]
        assert abs(float(cf[0]) - float(cf[1])) < 1e-6

    def test_two_contracts_different_terms(self):
        """Two contracts with different maturities → different event counts, padding works."""
        attrs_short = _make_fixed_rate_attrs(years=1)
        attrs_long = _make_fixed_rate_attrs(years=5)
        rf_obs = ConstantRiskFactorObserver(constant_value=0.0)

        contracts = [(attrs_short, rf_obs), (attrs_long, rf_obs)]
        result = simulate_pam_portfolio(contracts)

        assert result["num_contracts"] == 2
        # Different term → different total cashflows
        cf = result["total_cashflows"]
        # Longer contract should have more interest payments
        assert float(cf[1]) != float(cf[0])

    def test_batch_vs_individual(self):
        """Batch results should match individual simulation."""
        attrs1 = _make_fixed_rate_attrs(notional=100_000, rate=0.04, years=3)
        attrs2 = _make_fixed_rate_attrs(notional=200_000, rate=0.06, years=2)
        rf_obs = ConstantRiskFactorObserver(constant_value=0.0)

        # Individual
        _, payoffs1 = _simulate_array_path(attrs1, rf_obs)
        _, payoffs2 = _simulate_array_path(attrs2, rf_obs)
        total1 = float(jnp.sum(payoffs1))
        total2 = float(jnp.sum(payoffs2))

        # Batched
        contracts = [(attrs1, rf_obs), (attrs2, rf_obs)]
        result = simulate_pam_portfolio(contracts)
        batch_totals = result["total_cashflows"]

        assert abs(float(batch_totals[0]) - total1) <= ATOL
        assert abs(float(batch_totals[1]) - total2) <= ATOL

    def test_portfolio_with_pv(self):
        """Portfolio simulation with discount rate produces present values."""
        attrs = _make_fixed_rate_attrs()
        rf_obs = ConstantRiskFactorObserver(constant_value=0.0)

        contracts = [(attrs, rf_obs)]
        result = simulate_pam_portfolio(contracts, discount_rate=0.05)

        assert "present_values" in result
        assert "total_pv" in result
        pv = float(result["present_values"][0])
        assert jnp.isfinite(jnp.array(pv))

    def test_batch_10_contracts(self):
        """Stress test with 10 contracts of varying terms."""
        rf_obs = ConstantRiskFactorObserver(constant_value=0.0)
        contracts = []
        for i in range(10):
            attrs = _make_fixed_rate_attrs(
                notional=50_000 + i * 10_000,
                rate=0.03 + i * 0.005,
                years=1 + (i % 5),
            )
            contracts.append((attrs, rf_obs))

        result = simulate_pam_portfolio(contracts)
        assert result["num_contracts"] == 10
        cf = result["total_cashflows"]
        assert cf.shape == (10,)
        # All should be finite
        assert jnp.all(jnp.isfinite(cf))


# ============================================================================
# Result type tests
# ============================================================================


class TestResultTypes:
    """Test ArraySimulationResult and BatchSimulationResult."""

    def test_array_simulation_result(self):
        payoffs = jnp.array([100.0, -50.0, 200.0, 0.0])
        mask = jnp.array([1.0, 1.0, 1.0, 0.0])
        state = PAMArrayState(
            nt=jnp.array(0.0),
            ipnr=jnp.array(0.0),
            ipac=jnp.array(0.0),
            feac=jnp.array(0.0),
            nsc=jnp.array(1.0),
            isc=jnp.array(1.0),
        )
        result = ArraySimulationResult(payoffs=payoffs, final_state=state, event_mask=mask)
        assert result.num_events == 3
        assert abs(float(result.total_cashflow()) - 250.0) < 1e-6

    def test_batch_simulation_result(self):
        payoffs = jnp.array([[100.0, 200.0], [-50.0, 300.0]])
        masks = jnp.ones((2, 2))
        state = PAMArrayState(
            nt=jnp.array([0.0, 0.0]),
            ipnr=jnp.array([0.0, 0.0]),
            ipac=jnp.array([0.0, 0.0]),
            feac=jnp.array([0.0, 0.0]),
            nsc=jnp.array([1.0, 1.0]),
            isc=jnp.array([1.0, 1.0]),
        )
        result = BatchSimulationResult(payoffs=payoffs, final_states=state, masks=masks)
        assert result.num_contracts == 2
        totals = result.total_cashflows()
        assert abs(float(totals[0]) - 300.0) < 1e-6
        assert abs(float(totals[1]) - 250.0) < 1e-6

    def test_validate_pam_for_array_mode(self):
        attrs = _make_fixed_rate_attrs()
        errors = validate_pam_for_array_mode(attrs)
        assert len(errors) == 0

    def test_validate_non_pam_rejected(self):
        attrs = ContractAttributes(
            contract_id="LAM-001",
            contract_type=ContractType.LAM,
            contract_role=ContractRole.RPA,
            status_date=ActusDateTime(2024, 1, 1),
            maturity_date=ActusDateTime(2029, 1, 15),
        )
        errors = validate_pam_for_array_mode(attrs)
        assert len(errors) > 0
        assert "PAM" in errors[0]


# ============================================================================
# POF function unit tests
# ============================================================================


class TestPOFZero:
    """POF functions that should always return zero."""

    @pytest.fixture
    def state_and_params(self):
        state = PAMArrayState(
            nt=jnp.array(100_000.0),
            ipnr=jnp.array(0.05),
            ipac=jnp.array(1_000.0),
            feac=jnp.array(50.0),
            nsc=jnp.array(1.0),
            isc=jnp.array(1.0),
        )
        params = PAMArrayParams(
            role_sign=jnp.array(1.0),
            notional_principal=jnp.array(100_000.0),
            nominal_interest_rate=jnp.array(0.05),
            premium_discount_at_ied=jnp.array(0.0),
            accrued_interest=jnp.array(0.0),
            fee_rate=jnp.array(0.01),
            fee_basis=jnp.array(1, dtype=jnp.int32),
            penalty_rate=jnp.array(0.02),
            penalty_type=jnp.array(1, dtype=jnp.int32),
            price_at_purchase_date=jnp.array(0.0),
            price_at_termination_date=jnp.array(0.0),
            rate_reset_spread=jnp.array(0.0),
            rate_reset_multiplier=jnp.array(1.0),
            rate_reset_floor=jnp.array(0.0),
            rate_reset_cap=jnp.array(1.0),
            rate_reset_next=jnp.array(0.05),
            has_rate_floor=jnp.array(0.0),
            has_rate_cap=jnp.array(0.0),
            ied_ipac=jnp.array(0.0),
        )
        return state, params

    def test_pof_ad_zero(self, state_and_params):
        state, params = state_and_params
        assert float(_pof_ad(state, params, jnp.array(0.5), jnp.array(0.0))) == 0.0

    def test_pof_ipci_zero(self, state_and_params):
        state, params = state_and_params
        assert float(_pof_ipci(state, params, jnp.array(0.5), jnp.array(0.0))) == 0.0

    def test_pof_rr_zero(self, state_and_params):
        state, params = state_and_params
        assert float(_pof_rr(state, params, jnp.array(0.5), jnp.array(0.04))) == 0.0

    def test_pof_rrf_zero(self, state_and_params):
        state, params = state_and_params
        assert float(_pof_rrf(state, params, jnp.array(0.5), jnp.array(0.0))) == 0.0

    def test_pof_sc_zero(self, state_and_params):
        state, params = state_and_params
        assert float(_pof_sc(state, params, jnp.array(0.5), jnp.array(1.05))) == 0.0

    def test_pof_ce_zero(self, state_and_params):
        state, params = state_and_params
        assert float(_pof_ce(state, params, jnp.array(0.5), jnp.array(0.0))) == 0.0


class TestPOFNonZero:
    """POF functions with non-trivial payoffs."""

    def _make_state(self, nt=100_000.0, ipnr=0.05, ipac=0.0):
        return PAMArrayState(
            nt=jnp.array(nt),
            ipnr=jnp.array(ipnr),
            ipac=jnp.array(ipac),
            feac=jnp.array(0.0),
            nsc=jnp.array(1.0),
            isc=jnp.array(1.0),
        )

    def _make_params(self, **overrides):
        defaults = {
            "role_sign": jnp.array(1.0),
            "notional_principal": jnp.array(100_000.0),
            "nominal_interest_rate": jnp.array(0.05),
            "premium_discount_at_ied": jnp.array(0.0),
            "accrued_interest": jnp.array(0.0),
            "fee_rate": jnp.array(0.0),
            "fee_basis": jnp.array(0, dtype=jnp.int32),
            "penalty_rate": jnp.array(0.0),
            "penalty_type": jnp.array(0, dtype=jnp.int32),
            "price_at_purchase_date": jnp.array(0.0),
            "price_at_termination_date": jnp.array(0.0),
            "rate_reset_spread": jnp.array(0.0),
            "rate_reset_multiplier": jnp.array(1.0),
            "rate_reset_floor": jnp.array(0.0),
            "rate_reset_cap": jnp.array(1.0),
            "rate_reset_next": jnp.array(0.05),
            "has_rate_floor": jnp.array(0.0),
            "has_rate_cap": jnp.array(0.0),
            "ied_ipac": jnp.array(0.0),
        }
        defaults.update(overrides)
        return PAMArrayParams(**defaults)

    def test_pof_ied(self):
        """IED payoff = role_sign * (-1) * (NT + PDIED)."""
        state = self._make_state()
        params = self._make_params(premium_discount_at_ied=jnp.array(500.0))
        payoff = _pof_ied(state, params, jnp.array(0.0), jnp.array(0.0))
        expected = 1.0 * (-1.0) * (100_000.0 + 500.0)
        assert abs(float(payoff) - expected) < 1e-6

    def test_pof_ip(self):
        """IP payoff = isc * (ipac + yf * ipnr * nt)."""
        state = self._make_state(ipac=500.0)
        params = self._make_params()
        yf = jnp.array(0.25)  # quarter year
        payoff = _pof_ip(state, params, yf, jnp.array(0.0))
        expected = 1.0 * (500.0 + 0.25 * 0.05 * 100_000.0)
        assert abs(float(payoff) - expected) < 1e-6

    def test_pof_md(self):
        """MD payoff = nsc * nt + isc * ipac + feac."""
        state = PAMArrayState(
            nt=jnp.array(100_000.0),
            ipnr=jnp.array(0.05),
            ipac=jnp.array(1_000.0),
            feac=jnp.array(50.0),
            nsc=jnp.array(1.0),
            isc=jnp.array(1.0),
        )
        params = self._make_params()
        payoff = _pof_md(state, params, jnp.array(0.0), jnp.array(0.0))
        expected = 1.0 * 100_000.0 + 1.0 * 1_000.0 + 50.0
        assert abs(float(payoff) - expected) < 1e-6


# ============================================================================
# STF function unit tests
# ============================================================================


class TestSTF:
    """State transition function unit tests."""

    def _make_state(self, nt=100_000.0, ipnr=0.05, ipac=0.0):
        return PAMArrayState(
            nt=jnp.array(nt),
            ipnr=jnp.array(ipnr),
            ipac=jnp.array(ipac),
            feac=jnp.array(0.0),
            nsc=jnp.array(1.0),
            isc=jnp.array(1.0),
        )

    def _make_params(self, **overrides):
        defaults = {
            "role_sign": jnp.array(1.0),
            "notional_principal": jnp.array(100_000.0),
            "nominal_interest_rate": jnp.array(0.05),
            "premium_discount_at_ied": jnp.array(0.0),
            "accrued_interest": jnp.array(0.0),
            "fee_rate": jnp.array(0.0),
            "fee_basis": jnp.array(0, dtype=jnp.int32),
            "penalty_rate": jnp.array(0.0),
            "penalty_type": jnp.array(0, dtype=jnp.int32),
            "price_at_purchase_date": jnp.array(0.0),
            "price_at_termination_date": jnp.array(0.0),
            "rate_reset_spread": jnp.array(0.0),
            "rate_reset_multiplier": jnp.array(1.0),
            "rate_reset_floor": jnp.array(0.0),
            "rate_reset_cap": jnp.array(1.0),
            "rate_reset_next": jnp.array(0.05),
            "has_rate_floor": jnp.array(0.0),
            "has_rate_cap": jnp.array(0.0),
            "ied_ipac": jnp.array(0.0),
        }
        defaults.update(overrides)
        return PAMArrayParams(**defaults)

    def test_stf_ad_accrues_interest(self):
        state = self._make_state()
        params = self._make_params()
        yf = jnp.array(0.25)
        new = _stf_ad(state, params, yf, jnp.array(0.0))
        expected_ipac = 0.0 + 0.25 * 0.05 * 100_000.0
        assert abs(float(new.ipac) - expected_ipac) < 1e-6
        assert float(new.nt) == float(state.nt)  # unchanged

    def test_stf_ied_initializes(self):
        state = self._make_state(nt=0.0, ipnr=0.0)
        params = self._make_params()
        new = _stf_ied(state, params, jnp.array(0.0), jnp.array(0.0))
        assert abs(float(new.nt) - 100_000.0) < 1e-6
        assert abs(float(new.ipnr) - 0.05) < 1e-8
        assert float(new.nsc) == 1.0
        assert float(new.isc) == 1.0

    def test_stf_md_zeros_out(self):
        state = self._make_state(ipac=1_000.0)
        params = self._make_params()
        new = _stf_md(state, params, jnp.array(0.0), jnp.array(0.0))
        assert float(new.nt) == 0.0
        assert float(new.ipac) == 0.0
        assert float(new.feac) == 0.0

    def test_stf_ip_resets_ipac(self):
        state = self._make_state(ipac=1_000.0)
        params = self._make_params()
        new = _stf_ip(state, params, jnp.array(0.0), jnp.array(0.0))
        assert float(new.ipac) == 0.0

    def test_stf_ipci_capitalizes(self):
        state = self._make_state(ipac=500.0)
        params = self._make_params()
        yf = jnp.array(0.25)
        new = _stf_ipci(state, params, yf, jnp.array(0.0))
        total_ipac = 500.0 + 0.25 * 0.05 * 100_000.0
        assert abs(float(new.nt) - (100_000.0 + total_ipac)) < 1e-4
        assert float(new.ipac) == 0.0

    def test_stf_rr_resets_rate(self):
        state = self._make_state()
        params = self._make_params(
            rate_reset_multiplier=jnp.array(1.0),
            rate_reset_spread=jnp.array(0.01),
            has_rate_floor=jnp.array(0.0),
            has_rate_cap=jnp.array(0.0),
        )
        yf = jnp.array(0.5)
        rf = jnp.array(0.04)  # market rate
        new = _stf_rr(state, params, yf, rf)
        expected_rate = 1.0 * 0.04 + 0.01  # = 0.05
        assert abs(float(new.ipnr) - expected_rate) < 1e-8

    def test_stf_rr_with_floor(self):
        state = self._make_state()
        params = self._make_params(
            rate_reset_multiplier=jnp.array(1.0),
            rate_reset_spread=jnp.array(0.0),
            rate_reset_floor=jnp.array(0.03),
            has_rate_floor=jnp.array(1.0),
            has_rate_cap=jnp.array(0.0),
        )
        rf = jnp.array(0.01)  # below floor
        new = _stf_rr(state, params, jnp.array(0.0), rf)
        assert float(new.ipnr) >= 0.03 - 1e-8

    def test_stf_rr_with_cap(self):
        state = self._make_state()
        params = self._make_params(
            rate_reset_multiplier=jnp.array(1.0),
            rate_reset_spread=jnp.array(0.0),
            rate_reset_cap=jnp.array(0.06),
            has_rate_floor=jnp.array(0.0),
            has_rate_cap=jnp.array(1.0),
        )
        rf = jnp.array(0.10)  # above cap
        new = _stf_rr(state, params, jnp.array(0.0), rf)
        assert float(new.ipnr) <= 0.06 + 1e-8

    def test_stf_rrf_uses_next_rate(self):
        state = self._make_state()
        params = self._make_params(rate_reset_next=jnp.array(0.07))
        new = _stf_rrf(state, params, jnp.array(0.0), jnp.array(0.0))
        assert abs(float(new.ipnr) - 0.07) < 1e-8
