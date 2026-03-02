"""Equivalence tests for array-mode FXOUT simulation.

Runs identical contracts through both the Python path (FXOutrightContract)
and the array-mode path (simulate_fxout_array), asserting matching results.
Tolerance matches the ACTUS cross-validation standard (atol=1.0).
"""

import jax
import jax.numpy as jnp

from jactus.contracts.fxout import FXOutrightContract
from jactus.contracts.fxout_array import (
    FXOUTArrayParams,
    precompute_fxout_arrays,
    simulate_fxout_array,
    simulate_fxout_array_jit,
    simulate_fxout_portfolio,
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


def _make_fxout_gross_attrs(
    nt1: float = 100_000.0,
    nt2: float = 110_000.0,
    role: ContractRole = ContractRole.RPA,
) -> ContractAttributes:
    """Create FXOUT attributes with gross (delivery) settlement.

    DS='D' (delivery) with no settlement_period -> two MD events (gross).
    """
    return ContractAttributes(
        contract_id="FXOUT-GROSS",
        contract_type=ContractType.FXOUT,
        contract_role=role,
        status_date=ActusDateTime(2024, 1, 1),
        maturity_date=ActusDateTime(2024, 7, 1),
        currency="EUR",
        currency_2="USD",
        notional_principal=nt1,
        notional_principal_2=nt2,
        delivery_settlement="D",
    )


def _make_fxout_with_purchase_attrs(
    nt1: float = 100_000.0,
    nt2: float = 110_000.0,
    pprd: float = 500.0,
) -> ContractAttributes:
    """Create FXOUT attributes with a purchase event and gross settlement."""
    return ContractAttributes(
        contract_id="FXOUT-PRD",
        contract_type=ContractType.FXOUT,
        contract_role=ContractRole.RPA,
        status_date=ActusDateTime(2024, 1, 1),
        maturity_date=ActusDateTime(2024, 7, 1),
        currency="EUR",
        currency_2="USD",
        notional_principal=nt1,
        notional_principal_2=nt2,
        delivery_settlement="D",
        purchase_date=ActusDateTime(2024, 1, 15),
        price_at_purchase_date=pprd,
    )


def _simulate_python_path(attrs, rf_observer):
    """Run simulation through the standard Python path."""
    contract = FXOutrightContract(attrs, rf_observer)
    result = contract.simulate()
    return result


def _simulate_array_path(attrs, rf_observer):
    """Run simulation through the array-mode path."""
    arrays = precompute_fxout_arrays(attrs, rf_observer)
    initial_state, event_types, year_fractions, rf_values, params = arrays
    final_state, payoffs = simulate_fxout_array(
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
    """End-to-end equivalence: simulate_fxout_array vs contract.simulate()."""

    def test_gross_settlement(self):
        """FXOUT with gross (delivery) settlement -- two MD events."""
        attrs = _make_fxout_gross_attrs()
        rf_obs = ConstantRiskFactorObserver(constant_value=0.0)

        py_result = _simulate_python_path(attrs, rf_obs)
        _, payoffs = _simulate_array_path(attrs, rf_obs)
        _assert_payoffs_match(py_result, payoffs)

    def test_gross_settlement_rpl(self):
        """FXOUT from seller/borrower perspective (RPL)."""
        attrs = _make_fxout_gross_attrs(role=ContractRole.RPL)
        rf_obs = ConstantRiskFactorObserver(constant_value=0.0)

        py_result = _simulate_python_path(attrs, rf_obs)
        _, payoffs = _simulate_array_path(attrs, rf_obs)
        _assert_payoffs_match(py_result, payoffs)

    def test_with_purchase(self):
        """FXOUT with purchase event (PRD) and gross settlement."""
        attrs = _make_fxout_with_purchase_attrs()
        rf_obs = ConstantRiskFactorObserver(constant_value=0.0)

        py_result = _simulate_python_path(attrs, rf_obs)
        _, payoffs = _simulate_array_path(attrs, rf_obs)
        _assert_payoffs_match(py_result, payoffs)

    def test_net_settlement(self):
        """FXOUT with net cash settlement (DS='S' + settlement_period)."""
        attrs = ContractAttributes(
            contract_id="FXOUT-NET",
            contract_type=ContractType.FXOUT,
            contract_role=ContractRole.RPA,
            status_date=ActusDateTime(2024, 1, 1),
            maturity_date=ActusDateTime(2025, 1, 15),
            currency="USD",
            currency_2="EUR",
            notional_principal=100_000.0,
            notional_principal_2=90_000.0,
            delivery_settlement="S",
            settlement_period="P2D",
        )
        # FX rate: EUR/USD = 1.15, so net = 100000 - 1.15*90000 = -3500
        rf_obs = DictRiskFactorObserver({"EUR/USD": 1.15})

        py_result = _simulate_python_path(attrs, rf_obs)
        _, payoffs = _simulate_array_path(attrs, rf_obs)
        _assert_payoffs_match(py_result, payoffs)

    def test_net_settlement_rpl(self):
        """FXOUT net settlement from RPL (borrower/seller) perspective."""
        attrs = ContractAttributes(
            contract_id="FXOUT-NET-RPL",
            contract_type=ContractType.FXOUT,
            contract_role=ContractRole.RPL,
            status_date=ActusDateTime(2024, 1, 1),
            maturity_date=ActusDateTime(2025, 1, 15),
            currency="USD",
            currency_2="EUR",
            notional_principal=100_000.0,
            notional_principal_2=90_000.0,
            delivery_settlement="S",
            settlement_period="P2D",
        )
        # Same FX rate, but RPL flips the sign
        rf_obs = DictRiskFactorObserver({"EUR/USD": 1.15})

        py_result = _simulate_python_path(attrs, rf_obs)
        _, payoffs = _simulate_array_path(attrs, rf_obs)
        _assert_payoffs_match(py_result, payoffs)

    def test_total_cashflow_equivalence(self):
        """Total cashflow should match between paths."""
        attrs = _make_fxout_gross_attrs(nt1=100_000.0, nt2=110_000.0)
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
    """Batch FXOUT simulation equivalence tests."""

    def test_batch_matches_individual(self):
        """Batch simulation should match individual simulations."""
        rf_obs = ConstantRiskFactorObserver(0.0)
        contracts = [
            (_make_fxout_gross_attrs(nt1=100_000.0, nt2=110_000.0), rf_obs),
            (_make_fxout_gross_attrs(nt1=50_000.0, nt2=55_000.0, role=ContractRole.RPL), rf_obs),
            (_make_fxout_with_purchase_attrs(nt1=80_000.0, nt2=88_000.0, pprd=200.0), rf_obs),
        ]

        # Individual simulations
        individual_totals = []
        for attrs, obs in contracts:
            _, payoffs = _simulate_array_path(attrs, obs)
            individual_totals.append(float(jnp.sum(payoffs)))

        # Batch simulation
        result = simulate_fxout_portfolio(contracts)
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
        attrs = _make_fxout_gross_attrs()
        rf_obs = ConstantRiskFactorObserver(0.0)

        arrays = precompute_fxout_arrays(attrs, rf_obs)
        _, payoffs_eager = simulate_fxout_array(*arrays)
        _, payoffs_jit = simulate_fxout_array_jit(*arrays)

        assert jnp.allclose(payoffs_eager, payoffs_jit, atol=1e-6)


# ============================================================================
# Gradient tests
# ============================================================================


class TestGradients:
    """Test that gradients can be computed through FXOUT simulation."""

    def test_gradient_wrt_notional_1(self):
        """dTotal/dNT1 should be role_sign (since gross settlement pays +rs*NT1)."""
        attrs = _make_fxout_gross_attrs()
        rf_obs = ConstantRiskFactorObserver(0.0)

        arrays = precompute_fxout_arrays(attrs, rf_obs)
        state, et, yf, rf, params = arrays

        def total_cashflow(nt1_val):
            p = FXOUTArrayParams(
                role_sign=params.role_sign,
                pprd=params.pprd,
                ptd=params.ptd,
                notional_1=nt1_val,
                notional_2=params.notional_2,
            )
            _, payoffs = simulate_fxout_array(state, et, yf, rf, p)
            return jnp.sum(payoffs)

        grad = jax.grad(total_cashflow)(params.notional_1)
        assert jnp.isfinite(grad)
        assert float(grad) != 0.0
