"""Equivalence tests for array-mode OPTNS simulation.

Runs identical contracts through both the Python path (OptionContract)
and the array-mode path (simulate_optns_array), asserting matching results.
Tolerance matches the ACTUS cross-validation standard (atol=1.0).
"""

import jax
import jax.numpy as jnp
import pytest

from jactus.contracts.optns import OptionContract
from jactus.contracts.optns_array import (
    OPTNSArrayParams,
    OPTNSArrayState,
    precompute_optns_arrays,
    simulate_optns_array,
    simulate_optns_array_jit,
    simulate_optns_portfolio,
)
from jactus.core import (
    ActusDateTime,
    ContractAttributes,
    ContractRole,
    ContractType,
)
from jactus.observers import (
    ConstantRiskFactorObserver,
    DictRiskFactorObserver,
    TimeSeriesRiskFactorObserver,
)

ATOL = 1.0


# ============================================================================
# Fixtures
# ============================================================================


def _make_call_option_attrs(
    strike: float = 100.0,
    role: ContractRole = ContractRole.RPA,
    spot_ref: str = "SPX",
) -> ContractAttributes:
    """Create European call option attributes (no premium)."""
    return ContractAttributes(
        contract_id="OPT-CALL",
        contract_type=ContractType.OPTNS,
        contract_role=role,
        status_date=ActusDateTime(2024, 1, 1),
        maturity_date=ActusDateTime(2024, 12, 31),
        currency="USD",
        option_type="C",
        option_strike_1=strike,
        option_exercise_type="E",
        contract_structure=spot_ref,
    )


def _make_put_option_attrs(
    strike: float = 100.0,
    role: ContractRole = ContractRole.RPA,
    spot_ref: str = "SPX",
) -> ContractAttributes:
    """Create European put option attributes (no premium)."""
    return ContractAttributes(
        contract_id="OPT-PUT",
        contract_type=ContractType.OPTNS,
        contract_role=role,
        status_date=ActusDateTime(2024, 1, 1),
        maturity_date=ActusDateTime(2024, 12, 31),
        currency="USD",
        option_type="P",
        option_strike_1=strike,
        option_exercise_type="E",
        contract_structure=spot_ref,
    )


def _make_option_with_premium_attrs(
    strike: float = 100.0,
    pprd: float = 5.0,
    option_type: str = "C",
    spot_ref: str = "SPX",
) -> ContractAttributes:
    """Create European option with premium (purchase event)."""
    return ContractAttributes(
        contract_id="OPT-PREMIUM",
        contract_type=ContractType.OPTNS,
        contract_role=ContractRole.RPA,
        status_date=ActusDateTime(2024, 1, 1),
        maturity_date=ActusDateTime(2024, 12, 31),
        currency="USD",
        option_type=option_type,
        option_strike_1=strike,
        option_exercise_type="E",
        contract_structure=spot_ref,
        purchase_date=ActusDateTime(2024, 1, 15),
        price_at_purchase_date=pprd,
    )


def _simulate_python_path(attrs, rf_observer):
    """Run simulation through the standard Python path."""
    contract = OptionContract(attrs, rf_observer)
    result = contract.simulate()
    return result


def _simulate_array_path(attrs, rf_observer):
    """Run simulation through the array-mode path."""
    arrays = precompute_optns_arrays(attrs, rf_observer)
    initial_state, event_types, year_fractions, rf_values, params = arrays
    final_state, payoffs = simulate_optns_array(
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
    """End-to-end equivalence: simulate_optns_array vs contract.simulate()."""

    def test_call_itm(self):
        """In-the-money call option (spot > strike)."""
        attrs = _make_call_option_attrs(strike=100.0)
        rf_obs = DictRiskFactorObserver({"SPX": 110.0})

        py_result = _simulate_python_path(attrs, rf_obs)
        _, payoffs = _simulate_array_path(attrs, rf_obs)
        _assert_payoffs_match(py_result, payoffs)

    def test_put_itm(self):
        """In-the-money put option (spot < strike)."""
        attrs = _make_put_option_attrs(strike=100.0)
        rf_obs = DictRiskFactorObserver({"SPX": 90.0})

        py_result = _simulate_python_path(attrs, rf_obs)
        _, payoffs = _simulate_array_path(attrs, rf_obs)
        _assert_payoffs_match(py_result, payoffs)

    def test_call_otm(self):
        """Out-of-the-money call option (spot < strike, payoff=0)."""
        attrs = _make_call_option_attrs(strike=100.0)
        rf_obs = DictRiskFactorObserver({"SPX": 90.0})

        py_result = _simulate_python_path(attrs, rf_obs)
        _, payoffs = _simulate_array_path(attrs, rf_obs)
        _assert_payoffs_match(py_result, payoffs)

    def test_with_premium(self):
        """Option with premium payment (purchase_date + price_at_purchase_date)."""
        attrs = _make_option_with_premium_attrs(strike=100.0, pprd=5.0)
        rf_obs = DictRiskFactorObserver({"SPX": 110.0})

        py_result = _simulate_python_path(attrs, rf_obs)
        _, payoffs = _simulate_array_path(attrs, rf_obs)
        _assert_payoffs_match(py_result, payoffs)

    def test_american_call_itm(self):
        """American call option with time-varying spot price (spot > strike)."""
        attrs = ContractAttributes(
            contract_id="OPTNS-AM-C",
            contract_type=ContractType.OPTNS,
            contract_role=ContractRole.RPA,
            status_date=ActusDateTime(2024, 1, 1),
            maturity_date=ActusDateTime(2025, 1, 15),
            currency="USD",
            notional_principal=100.0,
            option_strike_1=100.0,
            option_type="C",
            option_exercise_type="A",
            purchase_date=ActusDateTime(2024, 1, 15),
            price_at_purchase_date=10.0,
            contract_structure="SPX",
        )
        ts_data = {
            "SPX": [
                (ActusDateTime(2024, 1, 1), 95.0),
                (ActusDateTime(2024, 4, 1), 105.0),
                (ActusDateTime(2024, 7, 1), 115.0),
                (ActusDateTime(2024, 10, 1), 108.0),
                (ActusDateTime(2025, 1, 1), 112.0),
            ]
        }
        rf_obs = TimeSeriesRiskFactorObserver(risk_factors=ts_data)

        py_result = _simulate_python_path(attrs, rf_obs)
        _, payoffs = _simulate_array_path(attrs, rf_obs)

        # Compare total cashflows (American options have multiple XD events)
        py_total = sum(float(e.payoff) for e in py_result.events)
        array_total = float(jnp.sum(payoffs))
        assert abs(array_total - py_total) <= ATOL

    def test_american_put_itm(self):
        """American put option with time-varying spot price (spot < strike)."""
        attrs = ContractAttributes(
            contract_id="OPTNS-AM-P",
            contract_type=ContractType.OPTNS,
            contract_role=ContractRole.RPA,
            status_date=ActusDateTime(2024, 1, 1),
            maturity_date=ActusDateTime(2025, 1, 15),
            currency="USD",
            notional_principal=100.0,
            option_strike_1=100.0,
            option_type="P",
            option_exercise_type="A",
            purchase_date=ActusDateTime(2024, 1, 15),
            price_at_purchase_date=8.0,
            contract_structure="SPX",
        )
        ts_data = {
            "SPX": [
                (ActusDateTime(2024, 1, 1), 105.0),
                (ActusDateTime(2024, 4, 1), 95.0),
                (ActusDateTime(2024, 7, 1), 88.0),
                (ActusDateTime(2024, 10, 1), 92.0),
                (ActusDateTime(2025, 1, 1), 90.0),
            ]
        }
        rf_obs = TimeSeriesRiskFactorObserver(risk_factors=ts_data)

        py_result = _simulate_python_path(attrs, rf_obs)
        _, payoffs = _simulate_array_path(attrs, rf_obs)

        py_total = sum(float(e.payoff) for e in py_result.events)
        array_total = float(jnp.sum(payoffs))
        assert abs(array_total - py_total) <= ATOL

    def test_sel_role(self):
        """Option from seller perspective (SEL) -- signs flipped."""
        attrs = ContractAttributes(
            contract_id="OPTNS-SEL",
            contract_type=ContractType.OPTNS,
            contract_role=ContractRole.SEL,
            status_date=ActusDateTime(2024, 1, 1),
            maturity_date=ActusDateTime(2025, 1, 15),
            currency="USD",
            notional_principal=100.0,
            option_strike_1=100.0,
            option_type="C",
            option_exercise_type="E",
            contract_structure="SPX",
        )
        rf_obs = DictRiskFactorObserver({"SPX": 110.0})

        py_result = _simulate_python_path(attrs, rf_obs)
        _, payoffs = _simulate_array_path(attrs, rf_obs)
        _assert_payoffs_match(py_result, payoffs)

    def test_total_cashflow_equivalence(self):
        """Total cashflow should match between paths."""
        attrs = _make_option_with_premium_attrs(strike=100.0, pprd=5.0)
        rf_obs = DictRiskFactorObserver({"SPX": 115.0})

        py_result = _simulate_python_path(attrs, rf_obs)
        _, payoffs = _simulate_array_path(attrs, rf_obs)

        py_total = sum(float(e.payoff) for e in py_result.events)
        array_total = float(jnp.sum(payoffs))
        assert abs(array_total - py_total) <= ATOL


# ============================================================================
# Batch tests
# ============================================================================


class TestBatchEquivalence:
    """Batch OPTNS simulation equivalence tests."""

    def test_batch_matches_individual(self):
        """Batch simulation should match individual simulations."""
        contracts = [
            (
                _make_call_option_attrs(strike=100.0),
                DictRiskFactorObserver({"SPX": 110.0}),
            ),
            (
                _make_put_option_attrs(strike=100.0),
                DictRiskFactorObserver({"SPX": 90.0}),
            ),
            (
                _make_option_with_premium_attrs(strike=120.0, pprd=3.0),
                DictRiskFactorObserver({"SPX": 130.0}),
            ),
        ]

        # Individual simulations
        individual_totals = []
        for attrs, obs in contracts:
            _, payoffs = _simulate_array_path(attrs, obs)
            individual_totals.append(float(jnp.sum(payoffs)))

        # Batch simulation
        result = simulate_optns_portfolio(contracts)
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
        attrs = _make_call_option_attrs(strike=100.0)
        rf_obs = DictRiskFactorObserver({"SPX": 110.0})

        arrays = precompute_optns_arrays(attrs, rf_obs)
        _, payoffs_eager = simulate_optns_array(*arrays)
        _, payoffs_jit = simulate_optns_array_jit(*arrays)

        assert jnp.allclose(payoffs_eager, payoffs_jit, atol=1e-6)


# ============================================================================
# Gradient tests
# ============================================================================


class TestGradients:
    """Test that gradients can be computed through OPTNS simulation."""

    def test_gradient_wrt_pprd(self):
        """dTotal/dPPRD should be finite and non-zero for option with premium."""
        attrs = _make_option_with_premium_attrs(strike=100.0, pprd=5.0)
        rf_obs = DictRiskFactorObserver({"SPX": 110.0})

        arrays = precompute_optns_arrays(attrs, rf_obs)
        state, et, yf, rf, params = arrays

        def total_cashflow(pprd_val):
            p = OPTNSArrayParams(
                role_sign=params.role_sign,
                pprd=pprd_val,
                ptd=params.ptd,
            )
            _, payoffs = simulate_optns_array(state, et, yf, rf, p)
            return jnp.sum(payoffs)

        grad = jax.grad(total_cashflow)(params.pprd)
        assert jnp.isfinite(grad)
        assert float(grad) != 0.0
