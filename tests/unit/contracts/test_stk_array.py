"""Equivalence tests for array-mode STK simulation.

Runs identical contracts through both the Python path (StockContract)
and the array-mode path (simulate_stk_array), asserting matching results.
Tolerance matches the ACTUS cross-validation standard (atol=1.0).
"""

import jax.numpy as jnp

from jactus.contracts.stk import StockContract
from jactus.contracts.stk_array import (
    STKArrayParams,
    precompute_stk_arrays,
    simulate_stk_array,
    simulate_stk_array_jit,
    simulate_stk_portfolio,
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


def _make_stk_attrs_prd_td(
    pprd: float = 150.0,
    ptd: float = 180.0,
    role: ContractRole = ContractRole.RPA,
) -> ContractAttributes:
    """Create STK attributes with purchase and termination events."""
    return ContractAttributes(
        contract_id="STK-001",
        contract_type=ContractType.STK,
        contract_role=role,
        status_date=ActusDateTime(2024, 1, 1),
        currency="USD",
        purchase_date=ActusDateTime(2024, 1, 15),
        termination_date=ActusDateTime(2025, 1, 15),
        price_at_purchase_date=pprd,
        price_at_termination_date=ptd,
    )


def _make_stk_attrs_with_dividends(
    pprd: float = 150.0,
    ptd: float = 180.0,
    dv_amount: float = 2.0,
) -> ContractAttributes:
    """Create STK attributes with quarterly dividend payments."""
    return ContractAttributes(
        contract_id="STK-002",
        contract_type=ContractType.STK,
        contract_role=ContractRole.RPA,
        status_date=ActusDateTime(2024, 1, 1),
        currency="USD",
        purchase_date=ActusDateTime(2024, 1, 15),
        termination_date=ActusDateTime(2025, 1, 15),
        price_at_purchase_date=pprd,
        price_at_termination_date=ptd,
        dividend_cycle="3M",
        dividend_anchor=ActusDateTime(2024, 3, 15),
        market_object_code_of_dividends="AAPL-DIV",
    )


def _make_stk_attrs_prd_only(
    pprd: float = 150.0,
) -> ContractAttributes:
    """Create STK attributes with purchase only (no termination)."""
    return ContractAttributes(
        contract_id="STK-003",
        contract_type=ContractType.STK,
        contract_role=ContractRole.RPA,
        status_date=ActusDateTime(2024, 1, 1),
        currency="USD",
        purchase_date=ActusDateTime(2024, 1, 15),
        price_at_purchase_date=pprd,
    )


def _simulate_python_path(attrs, rf_observer):
    """Run simulation through the standard Python path."""
    contract = StockContract(attrs, rf_observer)
    result = contract.simulate()
    return result


def _simulate_array_path(attrs, rf_observer):
    """Run simulation through the array-mode path."""
    arrays = precompute_stk_arrays(attrs, rf_observer)
    initial_state, event_types, year_fractions, rf_values, params = arrays
    final_state, payoffs = simulate_stk_array(
        initial_state, event_types, year_fractions, rf_values, params
    )
    return final_state, payoffs


# ============================================================================
# End-to-end equivalence tests
# ============================================================================


class TestScanEquivalence:
    """End-to-end equivalence: simulate_stk_array vs contract.simulate()."""

    def test_prd_td_basic(self):
        """STK with purchase and termination — basic case."""
        attrs = _make_stk_attrs_prd_td()
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
        """STK from seller/borrower perspective (RPL)."""
        attrs = _make_stk_attrs_prd_td(role=ContractRole.RPL)
        rf_obs = ConstantRiskFactorObserver(constant_value=0.0)

        py_result = _simulate_python_path(attrs, rf_obs)
        _, payoffs = _simulate_array_path(attrs, rf_obs)

        py_payoffs = jnp.array([float(e.payoff) for e in py_result.events])
        for i in range(len(py_result.events)):
            assert abs(float(payoffs[i]) - float(py_payoffs[i])) <= ATOL, (
                f"Event {i} ({py_result.events[i].event_type.name}): "
                f"array={float(payoffs[i]):.2f}, python={float(py_payoffs[i]):.2f}"
            )

    def test_prd_payoff_is_negative(self):
        """Purchase payoff should be negative (outflow for buyer)."""
        attrs = _make_stk_attrs_prd_td(pprd=100.0, ptd=120.0)
        rf_obs = ConstantRiskFactorObserver(constant_value=0.0)

        _, payoffs = _simulate_array_path(attrs, rf_obs)
        # PRD is first event — should be negative
        assert float(payoffs[0]) < 0

    def test_td_payoff_is_positive(self):
        """Termination payoff should be positive (inflow for seller)."""
        attrs = _make_stk_attrs_prd_td(pprd=100.0, ptd=120.0)
        rf_obs = ConstantRiskFactorObserver(constant_value=0.0)

        _, payoffs = _simulate_array_path(attrs, rf_obs)
        # TD is last event — should be positive
        assert float(payoffs[-1]) > 0

    def test_with_dividends(self):
        """STK with quarterly dividend cycle."""
        attrs = _make_stk_attrs_with_dividends(dv_amount=2.0)
        dv_obs = DictRiskFactorObserver(risk_factors={"AAPL-DIV": 2.0})

        py_result = _simulate_python_path(attrs, dv_obs)
        _, payoffs = _simulate_array_path(attrs, dv_obs)

        py_payoffs = jnp.array([float(e.payoff) for e in py_result.events])
        assert py_payoffs.shape == payoffs.shape, (
            f"Event count mismatch: Python={py_payoffs.shape[0]}, Array={payoffs.shape[0]}"
        )
        for i in range(len(py_result.events)):
            assert abs(float(payoffs[i]) - float(py_payoffs[i])) <= ATOL, (
                f"Event {i} ({py_result.events[i].event_type.name}): "
                f"array={float(payoffs[i]):.2f}, python={float(py_payoffs[i]):.2f}"
            )

    def test_prd_only(self):
        """STK with purchase only — no termination date."""
        attrs = _make_stk_attrs_prd_only(pprd=150.0)
        rf_obs = ConstantRiskFactorObserver(constant_value=0.0)

        py_result = _simulate_python_path(attrs, rf_obs)
        _, payoffs = _simulate_array_path(attrs, rf_obs)

        py_payoffs = jnp.array([float(e.payoff) for e in py_result.events])
        assert py_payoffs.shape == payoffs.shape
        for i in range(len(py_result.events)):
            assert abs(float(payoffs[i]) - float(py_payoffs[i])) <= ATOL

    def test_total_cashflow_equivalence(self):
        """Total cashflow should match between paths."""
        attrs = _make_stk_attrs_prd_td(pprd=150.0, ptd=200.0)
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
    """Batch STK simulation equivalence tests."""

    def test_batch_matches_individual(self):
        """Batch simulation should match individual simulations."""
        rf_obs = ConstantRiskFactorObserver(0.0)
        contracts = [
            (_make_stk_attrs_prd_td(pprd=100.0, ptd=120.0), rf_obs),
            (_make_stk_attrs_prd_td(pprd=200.0, ptd=250.0, role=ContractRole.RPL), rf_obs),
            (_make_stk_attrs_prd_td(pprd=50.0, ptd=55.0), rf_obs),
        ]

        # Individual simulations
        individual_totals = []
        for attrs, obs in contracts:
            _, payoffs = _simulate_array_path(attrs, obs)
            individual_totals.append(float(jnp.sum(payoffs)))

        # Batch simulation
        result = simulate_stk_portfolio(contracts)
        batch_totals = result["total_cashflows"]

        for i in range(len(contracts)):
            assert abs(float(batch_totals[i]) - individual_totals[i]) <= ATOL

    def test_portfolio_net_cashflow(self):
        """Portfolio should correctly compute net cashflows."""
        rf_obs = ConstantRiskFactorObserver(0.0)
        contracts = [
            (_make_stk_attrs_prd_td(pprd=100.0, ptd=120.0), rf_obs),
            (_make_stk_attrs_prd_td(pprd=50.0, ptd=80.0), rf_obs),
        ]
        result = simulate_stk_portfolio(contracts)

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
        attrs = _make_stk_attrs_prd_td()
        rf_obs = ConstantRiskFactorObserver(0.0)

        arrays = precompute_stk_arrays(attrs, rf_obs)
        _, payoffs_eager = simulate_stk_array(*arrays)
        _, payoffs_jit = simulate_stk_array_jit(*arrays)

        assert jnp.allclose(payoffs_eager, payoffs_jit, atol=1e-6)


# ============================================================================
# Gradient tests
# ============================================================================


class TestGradients:
    """Test that gradients can be computed through STK simulation."""

    def test_gradient_wrt_pprd(self):
        """dTotal/dPPRD should be -role_sign (PRD payoff = -role_sign * pprd)."""
        import jax

        attrs = _make_stk_attrs_prd_td()
        rf_obs = ConstantRiskFactorObserver(0.0)

        arrays = precompute_stk_arrays(attrs, rf_obs)
        state, et, yf, rf, params = arrays

        def total_cashflow(pprd_val):
            p = STKArrayParams(
                role_sign=params.role_sign,
                pprd=pprd_val,
                ptd=params.ptd,
            )
            _, payoffs = simulate_stk_array(state, et, yf, rf, p)
            return jnp.sum(payoffs)

        grad = jax.grad(total_cashflow)(params.pprd)
        # dTotal/dPPRD = -role_sign (since PRD payoff = -role_sign * pprd)
        assert abs(float(grad) - (-1.0)) <= 1e-4  # role_sign=1 for RPA
