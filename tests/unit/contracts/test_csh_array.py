"""Equivalence tests for array-mode CSH simulation.

Runs identical contracts through both the Python path (CashContract)
and the array-mode path (simulate_csh_array), asserting matching results.
Tolerance matches the ACTUS cross-validation standard (atol=1.0).
"""

import jax.numpy as jnp

from jactus.contracts.csh import CashContract
from jactus.contracts.csh_array import (
    precompute_csh_arrays,
    simulate_csh_array,
    simulate_csh_array_jit,
    simulate_csh_portfolio,
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


def _make_csh_attrs(
    notional: float = 100_000.0,
    role: ContractRole = ContractRole.RPA,
) -> ContractAttributes:
    """Create attributes for a simple CSH contract."""
    return ContractAttributes(
        contract_id="CSH-001",
        contract_type=ContractType.CSH,
        contract_role=role,
        status_date=ActusDateTime(2024, 1, 1),
        currency="USD",
        notional_principal=notional,
    )


def _simulate_python_path(attrs, rf_observer):
    """Run simulation through the standard Python path."""
    contract = CashContract(attrs, rf_observer)
    result = contract.simulate()
    return result


def _simulate_array_path(attrs, rf_observer):
    """Run simulation through the array-mode path."""
    arrays = precompute_csh_arrays(attrs, rf_observer)
    initial_state, event_types, year_fractions, rf_values, params = arrays
    final_state, payoffs = simulate_csh_array(
        initial_state, event_types, year_fractions, rf_values, params
    )
    return final_state, payoffs


# ============================================================================
# End-to-end equivalence tests
# ============================================================================


class TestScanEquivalence:
    """End-to-end equivalence: simulate_csh_array vs contract.simulate()."""

    def test_basic_csh(self):
        """Basic CSH contract — single AD event with zero payoff."""
        attrs = _make_csh_attrs()
        rf_obs = ConstantRiskFactorObserver(constant_value=0.0)

        py_result = _simulate_python_path(attrs, rf_obs)
        final_state, payoffs = _simulate_array_path(attrs, rf_obs)

        py_payoffs = jnp.array([float(e.payoff) for e in py_result.events])
        # CSH has a single AD event with payoff=0.0
        assert payoffs.shape[0] >= 1
        for i in range(min(payoffs.shape[0], py_payoffs.shape[0])):
            assert abs(float(payoffs[i]) - float(py_payoffs[i])) <= ATOL

    def test_csh_rpl(self):
        """CSH from borrower perspective (RPL)."""
        attrs = _make_csh_attrs(role=ContractRole.RPL)
        rf_obs = ConstantRiskFactorObserver(constant_value=0.0)

        py_result = _simulate_python_path(attrs, rf_obs)
        final_state, payoffs = _simulate_array_path(attrs, rf_obs)

        py_payoffs = jnp.array([float(e.payoff) for e in py_result.events])
        for i in range(min(payoffs.shape[0], py_payoffs.shape[0])):
            assert abs(float(payoffs[i]) - float(py_payoffs[i])) <= ATOL

    def test_small_notional(self):
        """CSH with small notional — payoffs still zero."""
        attrs = _make_csh_attrs(notional=1.0)
        rf_obs = ConstantRiskFactorObserver(constant_value=0.0)

        _, payoffs = _simulate_array_path(attrs, rf_obs)
        assert float(jnp.sum(jnp.abs(payoffs))) == 0.0

    def test_all_payoffs_zero(self):
        """CSH payoffs should always be zero regardless of parameters."""
        attrs = _make_csh_attrs(notional=1_000_000.0)
        rf_obs = ConstantRiskFactorObserver(constant_value=100.0)

        _, payoffs = _simulate_array_path(attrs, rf_obs)
        assert float(jnp.sum(jnp.abs(payoffs))) == 0.0


# ============================================================================
# Batch tests
# ============================================================================


class TestBatchEquivalence:
    """Batch CSH simulation equivalence tests."""

    def test_batch_matches_individual(self):
        """Batch simulation should match individual simulations."""
        contracts = [
            (_make_csh_attrs(notional=100_000.0), ConstantRiskFactorObserver(0.0)),
            (_make_csh_attrs(notional=200_000.0), ConstantRiskFactorObserver(0.0)),
            (
                _make_csh_attrs(notional=50_000.0, role=ContractRole.RPL),
                ConstantRiskFactorObserver(0.0),
            ),
        ]

        # Individual simulations
        individual_payoffs = []
        for attrs, obs in contracts:
            _, payoffs = _simulate_array_path(attrs, obs)
            individual_payoffs.append(float(jnp.sum(payoffs)))

        # Batch simulation
        result = simulate_csh_portfolio(contracts)
        batch_totals = result["total_cashflows"]

        for i in range(len(contracts)):
            assert abs(float(batch_totals[i]) - individual_payoffs[i]) <= ATOL

    def test_portfolio_all_zeros(self):
        """CSH portfolio should always produce zero cashflows."""
        contracts = [
            (_make_csh_attrs(notional=n), ConstantRiskFactorObserver(0.0))
            for n in [100_000.0, 200_000.0, 300_000.0]
        ]
        result = simulate_csh_portfolio(contracts)
        assert float(jnp.sum(jnp.abs(result["total_cashflows"]))) == 0.0


# ============================================================================
# JIT and vmap tests
# ============================================================================


class TestJITCompilation:
    """Test JIT compilation works correctly."""

    def test_jit_matches_eager(self):
        """JIT-compiled version should match eager execution."""
        attrs = _make_csh_attrs()
        rf_obs = ConstantRiskFactorObserver(0.0)

        arrays = precompute_csh_arrays(attrs, rf_obs)
        _, payoffs_eager = simulate_csh_array(*arrays)
        _, payoffs_jit = simulate_csh_array_jit(*arrays)

        assert jnp.allclose(payoffs_eager, payoffs_jit, atol=1e-6)
