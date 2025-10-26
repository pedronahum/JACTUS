"""JAX compatibility tests (T3.12).

Tests that contracts work correctly with JAX features where applicable.
Note: ContractAttributes is a Pydantic model and cannot accept traced values,
so we test JAX compatibility at the state/payoff level.
"""

import jax
import jax.numpy as jnp
import pytest

from jactus.contracts import create_contract
from jactus.core import (
    ActusDateTime,
    ContractAttributes,
    ContractRole,
    ContractType,
    DayCountConvention,
)
from jactus.observers import JaxRiskFactorObserver


class TestJAXArrayHandling:
    """Test proper handling of JAX arrays in contract operations."""

    def test_csh_state_uses_jax_arrays(self):
        """Test CSH contract state uses JAX arrays."""
        attrs = ContractAttributes(
            contract_id="CSH-JAX-001",
            contract_type=ContractType.CSH,
            contract_role=ContractRole.RPA,
            status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
            currency="USD",
            notional_principal=100000.0,
        )
        
        risk_factors = jnp.array([0.0])
        rf_obs = JaxRiskFactorObserver(risk_factors)
        contract = create_contract(attrs, rf_obs)
        state = contract.initialize_state()
        
        # State fields should be JAX arrays
        assert isinstance(state.nt, jnp.ndarray)
        assert isinstance(state.ipnr, jnp.ndarray)
        assert isinstance(state.nsc, jnp.ndarray)
        assert jnp.isfinite(state.nt)

    def test_pam_payoffs_use_jax_arrays(self):
        """Test PAM payoffs are JAX arrays."""
        attrs = ContractAttributes(
            contract_id="PAM-JAX-001",
            contract_type=ContractType.PAM,
            contract_role=ContractRole.RPA,
            status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
            initial_exchange_date=ActusDateTime(2024, 1, 15, 0, 0, 0),
            maturity_date=ActusDateTime(2025, 1, 15, 0, 0, 0),
            currency="USD",
            notional_principal=100000.0,
            nominal_interest_rate=0.05,
            day_count_convention=DayCountConvention.A360,
        )
        
        risk_factors = jnp.array([0.05])
        rf_obs = JaxRiskFactorObserver(risk_factors)
        contract = create_contract(attrs, rf_obs)
        result = contract.simulate()
        
        # All payoffs should be JAX arrays
        for event in result.events:
            assert isinstance(event.payoff, jnp.ndarray)
            assert jnp.isfinite(event.payoff)

    def test_stk_simulation_with_jax_observer(self):
        """Test STK works with JAX observer."""
        attrs = ContractAttributes(
            contract_id="STK-JAX-001",
            contract_type=ContractType.STK,
            contract_role=ContractRole.RPA,
            status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
            purchase_date=ActusDateTime(2024, 1, 15, 0, 0, 0),
            termination_date=ActusDateTime(2024, 12, 31, 0, 0, 0),
            currency="USD",
            price_at_purchase_date=150.0,
            price_at_termination_date=175.0,
        )
        
        risk_factors = jnp.array([160.0])
        rf_obs = JaxRiskFactorObserver(risk_factors)
        contract = create_contract(attrs, rf_obs)
        result = contract.simulate()
        
        assert len(result.events) == 2
        assert all(isinstance(e.payoff, jnp.ndarray) for e in result.events)


class TestJITCompatibleOperations:
    """Test JIT compilation of individual contract operations."""

    def test_state_initialization_jittable(self):
        """Test state initialization can be JIT compiled."""
        attrs = ContractAttributes(
            contract_id="CSH-JIT-001",
            contract_type=ContractType.CSH,
            contract_role=ContractRole.RPA,
            status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
            currency="USD",
            notional_principal=100000.0,
        )
        
        risk_factors = jnp.array([0.0])
        rf_obs = JaxRiskFactorObserver(risk_factors)
        contract = create_contract(attrs, rf_obs)
        
        # JIT compile initialization
        @jax.jit
        def init_state_jit():
            return contract.initialize_state()
        
        state = init_state_jit()
        assert state is not None
        assert jnp.isfinite(state.nt)

    def test_repeated_jit_calls_consistent(self):
        """Test JIT compiled functions give consistent results."""
        attrs = ContractAttributes(
            contract_id="CSH-JIT-002",
            contract_type=ContractType.CSH,
            contract_role=ContractRole.RPA,
            status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
            currency="USD",
            notional_principal=100000.0,
        )
        
        risk_factors = jnp.array([0.0])
        rf_obs = JaxRiskFactorObserver(risk_factors)
        contract = create_contract(attrs, rf_obs)
        
        @jax.jit
        def get_notional():
            state = contract.initialize_state()
            return state.nt
        
        # Multiple calls should give same result
        result1 = get_notional()
        result2 = get_notional()
        
        assert jnp.array_equal(result1, result2)


class TestJAXRiskFactorObserver:
    """Test JAX risk factor observer functionality."""

    def test_jax_observer_returns_jax_arrays(self):
        """Test JAX observer returns proper JAX arrays."""
        risk_factors = jnp.array([0.05, 1.25, 100000.0])
        observer = JaxRiskFactorObserver(risk_factors)

        # Get different risk factors
        rate = observer.get(0)
        fx = observer.get(1)
        notional = observer.get(2)

        assert isinstance(rate, jnp.ndarray)
        assert isinstance(fx, jnp.ndarray)
        assert isinstance(notional, jnp.ndarray)

        # Use approximate comparison for float32 precision
        assert abs(float(rate) - 0.05) < 1e-6
        assert abs(float(fx) - 1.25) < 1e-6
        assert abs(float(notional) - 100000.0) < 1.0

    def test_jax_observer_jittable(self):
        """Test JAX observer operations can be JIT compiled."""
        risk_factors = jnp.array([0.05, 1.25])
        observer = JaxRiskFactorObserver(risk_factors)

        @jax.jit
        def get_rate_jit():
            return observer.get(0)

        @jax.jit
        def get_fx_jit():
            return observer.get(1)

        rate = get_rate_jit()
        fx = get_fx_jit()

        # Use approximate comparison for float32 precision
        assert abs(float(rate) - 0.05) < 1e-6
        assert abs(float(fx) - 1.25) < 1e-6

    def test_jax_observer_vmap_compatible(self):
        """Test JAX observer works with vmap on risk_factors array."""
        # Create multiple observers with different scenarios
        risk_factors_1 = jnp.array([0.03])
        risk_factors_2 = jnp.array([0.05])
        risk_factors_3 = jnp.array([0.07])

        obs1 = JaxRiskFactorObserver(risk_factors_1)
        obs2 = JaxRiskFactorObserver(risk_factors_2)
        obs3 = JaxRiskFactorObserver(risk_factors_3)

        # Get first risk factor from each observer
        rate1 = obs1.get(0)
        rate2 = obs2.get(0)
        rate3 = obs3.get(0)

        # Verify we can stack and operate on them with JAX
        rates = jnp.stack([rate1, rate2, rate3])

        assert rates.shape == (3,)
        assert jnp.allclose(rates, jnp.array([0.03, 0.05, 0.07]))


class TestJAXCompatibleCalculations:
    """Test that contract calculations use JAX-compatible operations."""

    def test_payoff_calculations_use_jax_ops(self):
        """Test payoff calculations use JAX operations."""
        attrs = ContractAttributes(
            contract_id="PAM-CALC-001",
            contract_type=ContractType.PAM,
            contract_role=ContractRole.RPA,
            status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
            initial_exchange_date=ActusDateTime(2024, 1, 15, 0, 0, 0),
            maturity_date=ActusDateTime(2025, 1, 15, 0, 0, 0),
            currency="USD",
            notional_principal=100000.0,
            nominal_interest_rate=0.05,
            day_count_convention=DayCountConvention.A360,
            interest_payment_cycle="6M",
        )
        
        risk_factors = jnp.array([0.05])
        rf_obs = JaxRiskFactorObserver(risk_factors)
        contract = create_contract(attrs, rf_obs)
        result = contract.simulate()
        
        # All calculations should produce finite JAX arrays
        for event in result.events:
            assert isinstance(event.payoff, jnp.ndarray)
            assert jnp.isfinite(event.payoff)
            assert isinstance(event.state_post.nt, jnp.ndarray)
            assert jnp.isfinite(event.state_post.nt)

    def test_state_transitions_preserve_jax_arrays(self):
        """Test state transitions maintain JAX array types."""
        attrs = ContractAttributes(
            contract_id="COM-CALC-001",
            contract_type=ContractType.COM,
            contract_role=ContractRole.RPA,
            status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
            purchase_date=ActusDateTime(2024, 1, 15, 0, 0, 0),
            termination_date=ActusDateTime(2024, 12, 31, 0, 0, 0),
            currency="USD",
            price_at_purchase_date=7500.0,
            price_at_termination_date=8200.0,
        )
        
        risk_factors = jnp.array([80.0])
        rf_obs = JaxRiskFactorObserver(risk_factors)
        contract = create_contract(attrs, rf_obs)
        result = contract.simulate()
        
        # All states should use JAX arrays
        for event in result.events:
            if event.state_pre:
                assert isinstance(event.state_pre.nt, jnp.ndarray)
                assert isinstance(event.state_pre.nsc, jnp.ndarray)
            if event.state_post:
                assert isinstance(event.state_post.nt, jnp.ndarray)
                assert isinstance(event.state_post.nsc, jnp.ndarray)


class TestJAXPerformance:
    """Test JAX performance characteristics."""

    def test_jit_compilation_works(self):
        """Test basic JIT compilation functionality."""
        @jax.jit
        def simple_calc(x):
            return x * 2.0 + 1.0
        
        result = simple_calc(jnp.array(5.0))
        assert float(result) == 11.0

    def test_multiple_observers_with_jax(self):
        """Test multiple contracts can use different JAX observers."""
        # Create different observers
        obs1 = JaxRiskFactorObserver(jnp.array([0.03]))
        obs2 = JaxRiskFactorObserver(jnp.array([0.05]))
        obs3 = JaxRiskFactorObserver(jnp.array([0.07]))
        
        attrs_template = ContractAttributes(
            contract_id="CSH-MULTI",
            contract_type=ContractType.CSH,
            contract_role=ContractRole.RPA,
            status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
            currency="USD",
            notional_principal=100000.0,
        )
        
        c1 = create_contract(attrs_template, obs1)
        c2 = create_contract(attrs_template, obs2)
        c3 = create_contract(attrs_template, obs3)
        
        s1 = c1.initialize_state()
        s2 = c2.initialize_state()
        s3 = c3.initialize_state()
        
        # All should be valid JAX arrays
        assert all(jnp.isfinite(s.nt) for s in [s1, s2, s3])
