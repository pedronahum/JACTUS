"""Equivalence tests for array-mode COM simulation.

Runs identical contracts through both the Python path (CommodityContract)
and the array-mode path (simulate_com_array), asserting matching results.
Tolerance matches the ACTUS cross-validation standard (atol=1.0).
"""

import jax.numpy as jnp
import pytest

from jactus.contracts.com import CommodityContract
from jactus.contracts.com_array import (
    COMArrayParams,
    COMArrayState,
    batch_simulate_com,
    precompute_com_arrays,
    prepare_com_batch,
    simulate_com_array,
    simulate_com_array_jit,
    simulate_com_portfolio,
)
from jactus.core import (
    ActusDateTime,
    ContractAttributes,
    ContractRole,
    ContractType,
)
from jactus.observers import ConstantRiskFactorObserver

ATOL = 1.0


# ============================================================================
# Fixtures
# ============================================================================


def _make_com_attrs_prd_td(
    pprd: float = 7500.0,
    ptd: float = 8200.0,
    quantity: float = 1.0,
    role: ContractRole = ContractRole.RPA,
) -> ContractAttributes:
    """Create COM attributes with purchase and termination events."""
    return ContractAttributes(
        contract_id="COM-001",
        contract_type=ContractType.COM,
        contract_role=role,
        status_date=ActusDateTime(2024, 1, 1),
        currency="USD",
        purchase_date=ActusDateTime(2024, 1, 15),
        termination_date=ActusDateTime(2025, 1, 15),
        price_at_purchase_date=pprd,
        price_at_termination_date=ptd,
        quantity=quantity,
    )


def _make_com_attrs_prd_only(
    pprd: float = 7500.0,
    quantity: float = 10.0,
) -> ContractAttributes:
    """Create COM attributes with purchase only."""
    return ContractAttributes(
        contract_id="COM-002",
        contract_type=ContractType.COM,
        contract_role=ContractRole.RPA,
        status_date=ActusDateTime(2024, 1, 1),
        currency="USD",
        purchase_date=ActusDateTime(2024, 1, 15),
        price_at_purchase_date=pprd,
        quantity=quantity,
    )


def _simulate_python_path(attrs, rf_observer):
    """Run simulation through the standard Python path."""
    contract = CommodityContract(attrs, rf_observer)
    result = contract.simulate()
    return result


def _simulate_array_path(attrs, rf_observer):
    """Run simulation through the array-mode path."""
    arrays = precompute_com_arrays(attrs, rf_observer)
    initial_state, event_types, year_fractions, rf_values, params = arrays
    final_state, payoffs = simulate_com_array(
        initial_state, event_types, year_fractions, rf_values, params
    )
    return final_state, payoffs


# ============================================================================
# End-to-end equivalence tests
# ============================================================================


class TestScanEquivalence:
    """End-to-end equivalence: simulate_com_array vs contract.simulate()."""

    def test_prd_td_basic(self):
        """COM with purchase and termination — basic case."""
        attrs = _make_com_attrs_prd_td()
        rf_obs = ConstantRiskFactorObserver(constant_value=0.0)

        py_result = _simulate_python_path(attrs, rf_obs)
        _, payoffs = _simulate_array_path(attrs, rf_obs)

        py_payoffs = jnp.array([float(e.payoff) for e in py_result.events])
        assert py_payoffs.shape == payoffs.shape, (
            f"Event count mismatch: Python={py_payoffs.shape[0]}, Array={payoffs.shape[0]}"
        )
        for i in range(len(py_result.events)):
            assert abs(float(payoffs[i]) - float(py_payoffs[i])) <= ATOL, (
                f"Event {i} ({py_result.events[i].event_type.name}): "
                f"array={float(payoffs[i]):.2f}, python={float(py_payoffs[i]):.2f}"
            )

    def test_prd_td_rpl(self):
        """COM from seller/borrower perspective (RPL)."""
        attrs = _make_com_attrs_prd_td(role=ContractRole.RPL)
        rf_obs = ConstantRiskFactorObserver(constant_value=0.0)

        py_result = _simulate_python_path(attrs, rf_obs)
        _, payoffs = _simulate_array_path(attrs, rf_obs)

        py_payoffs = jnp.array([float(e.payoff) for e in py_result.events])
        for i in range(len(py_result.events)):
            assert abs(float(payoffs[i]) - float(py_payoffs[i])) <= ATOL, (
                f"Event {i} ({py_result.events[i].event_type.name}): "
                f"array={float(payoffs[i]):.2f}, python={float(py_payoffs[i]):.2f}"
            )

    def test_with_quantity(self):
        """COM with quantity multiplier."""
        attrs = _make_com_attrs_prd_td(pprd=80.0, ptd=90.0, quantity=100.0)
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

    def test_prd_payoff_is_negative(self):
        """Purchase payoff should be negative (outflow for buyer)."""
        attrs = _make_com_attrs_prd_td(pprd=80.0, ptd=90.0)
        rf_obs = ConstantRiskFactorObserver(constant_value=0.0)

        _, payoffs = _simulate_array_path(attrs, rf_obs)
        # PRD is first event — should be negative
        assert float(payoffs[0]) < 0

    def test_td_payoff_is_positive(self):
        """Termination payoff should be positive (inflow for seller)."""
        attrs = _make_com_attrs_prd_td(pprd=80.0, ptd=90.0)
        rf_obs = ConstantRiskFactorObserver(constant_value=0.0)

        _, payoffs = _simulate_array_path(attrs, rf_obs)
        # TD is last event — should be positive
        assert float(payoffs[-1]) > 0

    def test_prd_only(self):
        """COM with purchase only — no termination date."""
        attrs = _make_com_attrs_prd_only()
        rf_obs = ConstantRiskFactorObserver(constant_value=0.0)

        py_result = _simulate_python_path(attrs, rf_obs)
        _, payoffs = _simulate_array_path(attrs, rf_obs)

        py_payoffs = jnp.array([float(e.payoff) for e in py_result.events])
        assert py_payoffs.shape == payoffs.shape
        for i in range(len(py_result.events)):
            assert abs(float(payoffs[i]) - float(py_payoffs[i])) <= ATOL

    def test_total_cashflow_equivalence(self):
        """Total cashflow should match between paths."""
        attrs = _make_com_attrs_prd_td(pprd=7500.0, ptd=8200.0, quantity=10.0)
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
    """Batch COM simulation equivalence tests."""

    def test_batch_matches_individual(self):
        """Batch simulation should match individual simulations."""
        rf_obs = ConstantRiskFactorObserver(0.0)
        contracts = [
            (_make_com_attrs_prd_td(pprd=7500.0, ptd=8200.0), rf_obs),
            (_make_com_attrs_prd_td(pprd=5000.0, ptd=5500.0, role=ContractRole.RPL), rf_obs),
            (_make_com_attrs_prd_td(pprd=3000.0, ptd=3100.0, quantity=50.0), rf_obs),
        ]

        # Individual simulations
        individual_totals = []
        for attrs, obs in contracts:
            _, payoffs = _simulate_array_path(attrs, obs)
            individual_totals.append(float(jnp.sum(payoffs)))

        # Batch simulation
        result = simulate_com_portfolio(contracts)
        batch_totals = result["total_cashflows"]

        for i in range(len(contracts)):
            assert abs(float(batch_totals[i]) - individual_totals[i]) <= ATOL

    def test_portfolio_net_cashflow(self):
        """Portfolio should correctly compute net cashflows."""
        rf_obs = ConstantRiskFactorObserver(0.0)
        contracts = [
            (_make_com_attrs_prd_td(pprd=100.0, ptd=120.0, quantity=1.0), rf_obs),
            (_make_com_attrs_prd_td(pprd=50.0, ptd=80.0, quantity=1.0), rf_obs),
        ]
        result = simulate_com_portfolio(contracts)

        # Contract 1: -100 + 120 = 20
        assert abs(float(result["total_cashflows"][0]) - 20.0) <= ATOL
        # Contract 2: -50 + 80 = 30
        assert abs(float(result["total_cashflows"][1]) - 30.0) <= ATOL


# ============================================================================
# JIT and vmap tests
# ============================================================================


class TestJITCompilation:
    """Test JIT compilation works correctly."""

    def test_jit_matches_eager(self):
        """JIT-compiled version should match eager execution."""
        attrs = _make_com_attrs_prd_td()
        rf_obs = ConstantRiskFactorObserver(0.0)

        arrays = precompute_com_arrays(attrs, rf_obs)
        _, payoffs_eager = simulate_com_array(*arrays)
        _, payoffs_jit = simulate_com_array_jit(*arrays)

        assert jnp.allclose(payoffs_eager, payoffs_jit, atol=1e-6)


# ============================================================================
# Gradient tests
# ============================================================================


class TestGradients:
    """Test that gradients can be computed through COM simulation."""

    def test_gradient_wrt_pprd(self):
        """dTotal/dPPRD should be -role_sign * quantity."""
        import jax

        attrs = _make_com_attrs_prd_td(quantity=10.0)
        rf_obs = ConstantRiskFactorObserver(0.0)

        arrays = precompute_com_arrays(attrs, rf_obs)
        state, et, yf, rf, params = arrays

        def total_cashflow(pprd_val):
            p = COMArrayParams(
                role_sign=params.role_sign,
                pprd=pprd_val,
                ptd=params.ptd,
                quantity=params.quantity,
            )
            _, payoffs = simulate_com_array(state, et, yf, rf, p)
            return jnp.sum(payoffs)

        grad = jax.grad(total_cashflow)(params.pprd)
        # dTotal/dPPRD = -role_sign * quantity = -1.0 * 10.0 = -10.0
        assert abs(float(grad) - (-10.0)) <= 1e-4

    def test_gradient_wrt_ptd(self):
        """dTotal/dPTD should be role_sign * quantity."""
        import jax

        attrs = _make_com_attrs_prd_td(quantity=5.0)
        rf_obs = ConstantRiskFactorObserver(0.0)

        arrays = precompute_com_arrays(attrs, rf_obs)
        state, et, yf, rf, params = arrays

        def total_cashflow(ptd_val):
            p = COMArrayParams(
                role_sign=params.role_sign,
                pprd=params.pprd,
                ptd=ptd_val,
                quantity=params.quantity,
            )
            _, payoffs = simulate_com_array(state, et, yf, rf, p)
            return jnp.sum(payoffs)

        grad = jax.grad(total_cashflow)(params.ptd)
        # dTotal/dPTD = role_sign * quantity = 1.0 * 5.0 = 5.0
        assert abs(float(grad) - 5.0) <= 1e-4
