"""Equivalence tests for array-mode FUTUR simulation.

Runs identical contracts through both the Python path (FutureContract)
and the array-mode path (simulate_futur_array), asserting matching results.
Tolerance matches the ACTUS cross-validation standard (atol=1.0).
"""

import jax
import jax.numpy as jnp
import pytest

from jactus.contracts.futur import FutureContract
from jactus.contracts.futur_array import (
    FUTURArrayParams,
    FUTURArrayState,
    precompute_futur_arrays,
    simulate_futur_array,
    simulate_futur_array_jit,
    simulate_futur_portfolio,
)
from jactus.core import (
    ActusDateTime,
    ContractAttributes,
    ContractRole,
    ContractType,
)
from jactus.observers import ConstantRiskFactorObserver, DictRiskFactorObserver

ATOL = 1.0


# ============================================================================
# Fixtures
# ============================================================================


def _make_futur_attrs(
    futures_price: float = 1800.0,
    notional: float = 100.0,
    role: ContractRole = ContractRole.RPA,
    spot_ref: str = "GC",
) -> ContractAttributes:
    """Create FUTUR attributes for a basic futures contract."""
    return ContractAttributes(
        contract_id="FUT-001",
        contract_type=ContractType.FUTUR,
        contract_role=role,
        status_date=ActusDateTime(2024, 1, 1),
        maturity_date=ActusDateTime(2024, 12, 31),
        currency="USD",
        notional_principal=notional,
        future_price=futures_price,
        contract_structure=spot_ref,
    )


def _make_futur_with_purchase_attrs(
    futures_price: float = 1800.0,
    notional: float = 100.0,
    pprd: float = 50.0,
    spot_ref: str = "GC",
) -> ContractAttributes:
    """Create FUTUR attributes with a purchase event."""
    return ContractAttributes(
        contract_id="FUT-PRD",
        contract_type=ContractType.FUTUR,
        contract_role=ContractRole.RPA,
        status_date=ActusDateTime(2024, 1, 1),
        maturity_date=ActusDateTime(2024, 12, 31),
        currency="USD",
        notional_principal=notional,
        future_price=futures_price,
        contract_structure=spot_ref,
        purchase_date=ActusDateTime(2024, 1, 15),
        price_at_purchase_date=pprd,
    )


def _simulate_python_path(attrs, rf_observer):
    """Run simulation through the standard Python path."""
    contract = FutureContract(attrs, rf_observer)
    result = contract.simulate()
    return result


def _simulate_array_path(attrs, rf_observer):
    """Run simulation through the array-mode path."""
    arrays = precompute_futur_arrays(attrs, rf_observer)
    initial_state, event_types, year_fractions, rf_values, params = arrays
    final_state, payoffs = simulate_futur_array(
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
    """End-to-end equivalence: simulate_futur_array vs contract.simulate()."""

    def test_basic_long(self):
        """Basic long futures position (RPA), spot > futures price."""
        attrs = _make_futur_attrs(futures_price=1800.0, notional=100.0)
        rf_obs = DictRiskFactorObserver({"GC": 1850.0})

        py_result = _simulate_python_path(attrs, rf_obs)
        _, payoffs = _simulate_array_path(attrs, rf_obs)
        _assert_payoffs_match(py_result, payoffs)

    def test_short_position(self):
        """Short futures position (RPL)."""
        attrs = _make_futur_attrs(
            futures_price=1800.0, notional=100.0, role=ContractRole.RPL
        )
        rf_obs = DictRiskFactorObserver({"GC": 1850.0})

        py_result = _simulate_python_path(attrs, rf_obs)
        _, payoffs = _simulate_array_path(attrs, rf_obs)
        _assert_payoffs_match(py_result, payoffs)

    def test_with_purchase(self):
        """FUTUR with purchase event (PRD)."""
        attrs = _make_futur_with_purchase_attrs(
            futures_price=1800.0, notional=100.0, pprd=50.0
        )
        rf_obs = DictRiskFactorObserver({"GC": 1850.0})

        py_result = _simulate_python_path(attrs, rf_obs)
        _, payoffs = _simulate_array_path(attrs, rf_obs)
        _assert_payoffs_match(py_result, payoffs)

    def test_total_cashflow_equivalence(self):
        """Total cashflow should match between paths."""
        attrs = _make_futur_attrs(futures_price=1800.0, notional=50.0)
        rf_obs = DictRiskFactorObserver({"GC": 1900.0})

        py_result = _simulate_python_path(attrs, rf_obs)
        _, payoffs = _simulate_array_path(attrs, rf_obs)

        py_total = sum(float(e.payoff) for e in py_result.events)
        array_total = float(jnp.sum(payoffs))
        assert abs(array_total - py_total) <= ATOL


# ============================================================================
# Batch tests
# ============================================================================


class TestBatchEquivalence:
    """Batch FUTUR simulation equivalence tests."""

    def test_batch_matches_individual(self):
        """Batch simulation should match individual simulations."""
        contracts = [
            (
                _make_futur_attrs(futures_price=1800.0, notional=100.0),
                DictRiskFactorObserver({"GC": 1850.0}),
            ),
            (
                _make_futur_attrs(
                    futures_price=1800.0, notional=50.0, role=ContractRole.RPL
                ),
                DictRiskFactorObserver({"GC": 1900.0}),
            ),
            (
                _make_futur_with_purchase_attrs(
                    futures_price=2000.0, notional=10.0, pprd=25.0
                ),
                DictRiskFactorObserver({"GC": 2100.0}),
            ),
        ]

        # Individual simulations
        individual_totals = []
        for attrs, obs in contracts:
            _, payoffs = _simulate_array_path(attrs, obs)
            individual_totals.append(float(jnp.sum(payoffs)))

        # Batch simulation
        result = simulate_futur_portfolio(contracts)
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
        attrs = _make_futur_attrs()
        rf_obs = DictRiskFactorObserver({"GC": 1850.0})

        arrays = precompute_futur_arrays(attrs, rf_obs)
        _, payoffs_eager = simulate_futur_array(*arrays)
        _, payoffs_jit = simulate_futur_array_jit(*arrays)

        assert jnp.allclose(payoffs_eager, payoffs_jit, atol=1e-6)


# ============================================================================
# Gradient tests
# ============================================================================


class TestGradients:
    """Test that gradients can be computed through FUTUR simulation."""

    def test_gradient_wrt_pprd(self):
        """dTotal/dPPRD should be finite and non-zero for a contract with purchase."""
        attrs = _make_futur_with_purchase_attrs(pprd=50.0)
        rf_obs = DictRiskFactorObserver({"GC": 1850.0})

        arrays = precompute_futur_arrays(attrs, rf_obs)
        state, et, yf, rf, params = arrays

        def total_cashflow(pprd_val):
            p = FUTURArrayParams(
                role_sign=params.role_sign,
                pprd=pprd_val,
                ptd=params.ptd,
                nt=params.nt,
            )
            _, payoffs = simulate_futur_array(state, et, yf, rf, p)
            return jnp.sum(payoffs)

        grad = jax.grad(total_cashflow)(params.pprd)
        assert jnp.isfinite(grad)
        assert float(grad) != 0.0
