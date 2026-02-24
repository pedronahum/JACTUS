"""Unit tests for risk factor observer.

T2.3: Risk Factor Observer Tests

Tests for:
- RiskFactorObserver Protocol
- BaseRiskFactorObserver ABC
- ConstantRiskFactorObserver
- DictRiskFactorObserver
- JAX compatibility
"""

import jax
import jax.numpy as jnp
import pytest

from jactus.core import ActusDateTime, ContractAttributes, ContractState
from jactus.core.types import ContractRole, ContractType, EventType
from jactus.observers import (
    BaseRiskFactorObserver,
    CallbackRiskFactorObserver,
    CompositeRiskFactorObserver,
    ConstantRiskFactorObserver,
    CurveRiskFactorObserver,
    DictRiskFactorObserver,
    JaxRiskFactorObserver,
    RiskFactorObserver,
    TimeSeriesRiskFactorObserver,
)


class TestRiskFactorObserverProtocol:
    """Test RiskFactorObserver protocol enforcement."""

    def test_protocol_requires_observe_risk_factor(self):
        """RiskFactorObserver protocol requires observe_risk_factor method."""

        class ValidObserver:
            def observe_risk_factor(self, identifier, time, state=None, attributes=None):
                return jnp.array(1.0)

            def observe_event(self, identifier, event_type, time, state=None, attributes=None):
                return None

        assert isinstance(ValidObserver(), RiskFactorObserver)

    def test_protocol_rejects_without_methods(self):
        """Objects without required methods are not RiskFactorObservers."""

        class InvalidObserver:
            pass

        assert not isinstance(InvalidObserver(), RiskFactorObserver)


class ConcreteRiskFactorObserver(BaseRiskFactorObserver):
    """Concrete implementation for testing BaseRiskFactorObserver."""

    def __init__(self, value: float = 1.0, name: str | None = None):
        super().__init__(name)
        self.value = jnp.array(value, dtype=jnp.float32)

    def _get_risk_factor(self, identifier, time, state, attributes):
        return self.value

    def _get_event_data(self, identifier, event_type, time, state, attributes):
        return self.value


class TestBaseRiskFactorObserverInit:
    """Test BaseRiskFactorObserver initialization."""

    def test_init_with_name(self):
        """Initialize with explicit name."""
        observer = ConcreteRiskFactorObserver(name="TestObserver")

        assert observer.name == "TestObserver"

    def test_init_without_name_uses_class_name(self):
        """Initialize without name uses class name."""
        observer = ConcreteRiskFactorObserver()

        assert observer.name == "ConcreteRiskFactorObserver"


class TestBaseRiskFactorObserverMethods:
    """Test BaseRiskFactorObserver methods."""

    def test_observe_risk_factor_calls_get_risk_factor(self):
        """observe_risk_factor calls _get_risk_factor."""
        observer = ConcreteRiskFactorObserver(value=1.5)
        time = ActusDateTime(2024, 6, 15, 0, 0, 0)

        result = observer.observe_risk_factor("TEST_RATE", time)

        assert jnp.allclose(result, jnp.array(1.5, dtype=jnp.float32))

    def test_observe_event_calls_get_event_data(self):
        """observe_event calls _get_event_data."""
        observer = ConcreteRiskFactorObserver(value=2.5)
        time = ActusDateTime(2024, 6, 15, 0, 0, 0)

        result = observer.observe_event("TEST_EVENT", EventType.IP, time)

        assert jnp.allclose(result, jnp.array(2.5, dtype=jnp.float32))


class TestConstantRiskFactorObserver:
    """Test ConstantRiskFactorObserver."""

    def test_init_with_value(self):
        """Initialize with constant value."""
        observer = ConstantRiskFactorObserver(1.18)

        assert jnp.allclose(observer.constant_value, jnp.array(1.18, dtype=jnp.float32))

    def test_observe_risk_factor_returns_constant(self):
        """Always returns constant value for any risk factor."""
        observer = ConstantRiskFactorObserver(0.05)
        time = ActusDateTime(2024, 6, 15, 0, 0, 0)

        result1 = observer.observe_risk_factor("USD/EUR", time)
        result2 = observer.observe_risk_factor("LIBOR-3M", time)
        result3 = observer.observe_risk_factor("ANYTHING", time)

        assert jnp.allclose(result1, jnp.array(0.05, dtype=jnp.float32))
        assert jnp.allclose(result2, jnp.array(0.05, dtype=jnp.float32))
        assert jnp.allclose(result3, jnp.array(0.05, dtype=jnp.float32))

    def test_observe_event_returns_constant(self):
        """Always returns constant value for any event."""
        observer = ConstantRiskFactorObserver(1.0)
        time = ActusDateTime(2024, 6, 15, 0, 0, 0)

        result1 = observer.observe_event("RESET_RATE", EventType.RR, time)
        result2 = observer.observe_event("OTHER_DATA", EventType.IP, time)

        assert jnp.allclose(result1, jnp.array(1.0, dtype=jnp.float32))
        assert jnp.allclose(result2, jnp.array(1.0, dtype=jnp.float32))

    def test_constant_value_ignores_state_and_attributes(self):
        """Constant value doesn't depend on state or attributes."""
        observer = ConstantRiskFactorObserver(3.14)
        time = ActusDateTime(2024, 6, 15, 0, 0, 0)

        tmd = ActusDateTime(2029, 1, 15, 0, 0, 0)
        sd = ActusDateTime(2024, 1, 15, 0, 0, 0)
        state = ContractState(
            tmd=tmd,
            sd=sd,
            nt=jnp.array(100000.0),
            ipnr=jnp.array(0.05),
            ipac=jnp.array(0.0),
            feac=jnp.array(0.0),
            nsc=jnp.array(1.0),
            isc=jnp.array(1.0),
        )

        attributes = ContractAttributes(
            contract_id="TEST001",
            contract_type=ContractType.PAM,
            contract_role=ContractRole.RPA,
            status_date=sd,
            currency="USD",
        )

        result = observer.observe_risk_factor("RATE", time, state=state, attributes=attributes)

        assert jnp.allclose(result, jnp.array(3.14, dtype=jnp.float32))


class TestDictRiskFactorObserver:
    """Test DictRiskFactorObserver."""

    def test_init_with_risk_factors(self):
        """Initialize with risk factors dictionary."""
        data = {"USD/EUR": 1.18, "LIBOR-3M": 0.02}
        observer = DictRiskFactorObserver(data)

        assert "USD/EUR" in observer.risk_factors
        assert "LIBOR-3M" in observer.risk_factors
        assert jnp.allclose(observer.risk_factors["USD/EUR"], jnp.array(1.18))
        assert jnp.allclose(observer.risk_factors["LIBOR-3M"], jnp.array(0.02))

    def test_init_with_event_data(self):
        """Initialize with event data dictionary."""
        risk_data = {"RATE": 0.05}
        event_data = {"RESET": 0.06}
        observer = DictRiskFactorObserver(risk_data, event_data=event_data)

        assert "RESET" in observer.event_data
        assert observer.event_data["RESET"] == 0.06

    def test_observe_risk_factor_returns_correct_value(self):
        """Returns correct value for known risk factor."""
        data = {"USD/EUR": 1.18, "GBP/USD": 1.25}
        observer = DictRiskFactorObserver(data)
        time = ActusDateTime(2024, 6, 15, 0, 0, 0)

        usd_eur = observer.observe_risk_factor("USD/EUR", time)
        gbp_usd = observer.observe_risk_factor("GBP/USD", time)

        assert jnp.allclose(usd_eur, jnp.array(1.18, dtype=jnp.float32))
        assert jnp.allclose(gbp_usd, jnp.array(1.25, dtype=jnp.float32))

    def test_observe_risk_factor_raises_keyerror_for_unknown(self):
        """Raises KeyError for unknown risk factor."""
        data = {"USD/EUR": 1.18}
        observer = DictRiskFactorObserver(data)
        time = ActusDateTime(2024, 6, 15, 0, 0, 0)

        with pytest.raises(KeyError) as exc_info:
            observer.observe_risk_factor("UNKNOWN_RATE", time)

        assert "UNKNOWN_RATE" in str(exc_info.value)
        assert "not found" in str(exc_info.value)

    def test_observe_event_returns_correct_value(self):
        """Returns correct value for known event data."""
        risk_data = {"RATE": 0.05}
        event_data = {"RESET_RATE": 0.06, "SPREAD": 0.01}
        observer = DictRiskFactorObserver(risk_data, event_data=event_data)
        time = ActusDateTime(2024, 6, 15, 0, 0, 0)

        reset_rate = observer.observe_event("RESET_RATE", EventType.RR, time)
        spread = observer.observe_event("SPREAD", EventType.IP, time)

        assert reset_rate == 0.06
        assert spread == 0.01

    def test_observe_event_raises_keyerror_for_unknown(self):
        """Raises KeyError for unknown event data."""
        observer = DictRiskFactorObserver({"RATE": 0.05})
        time = ActusDateTime(2024, 6, 15, 0, 0, 0)

        with pytest.raises(KeyError) as exc_info:
            observer.observe_event("UNKNOWN_EVENT", EventType.IP, time)

        assert "UNKNOWN_EVENT" in str(exc_info.value)
        assert "not found" in str(exc_info.value)

    def test_add_risk_factor(self):
        """Add new risk factor to observer."""
        observer = DictRiskFactorObserver({"USD/EUR": 1.18})
        time = ActusDateTime(2024, 6, 15, 0, 0, 0)

        observer.add_risk_factor("GBP/USD", 1.25)

        result = observer.observe_risk_factor("GBP/USD", time)
        assert jnp.allclose(result, jnp.array(1.25, dtype=jnp.float32))

    def test_add_risk_factor_updates_existing(self):
        """Update existing risk factor value."""
        observer = DictRiskFactorObserver({"USD/EUR": 1.18})
        time = ActusDateTime(2024, 6, 15, 0, 0, 0)

        observer.add_risk_factor("USD/EUR", 1.20)

        result = observer.observe_risk_factor("USD/EUR", time)
        assert jnp.allclose(result, jnp.array(1.20, dtype=jnp.float32))

    def test_add_event_data(self):
        """Add new event data to observer."""
        observer = DictRiskFactorObserver({"RATE": 0.05})
        time = ActusDateTime(2024, 6, 15, 0, 0, 0)

        observer.add_event_data("NEW_EVENT", "test_value")

        result = observer.observe_event("NEW_EVENT", EventType.IP, time)
        assert result == "test_value"

    def test_add_event_data_updates_existing(self):
        """Update existing event data value."""
        observer = DictRiskFactorObserver({"RATE": 0.05}, event_data={"EVENT": "old_value"})
        time = ActusDateTime(2024, 6, 15, 0, 0, 0)

        observer.add_event_data("EVENT", "new_value")

        result = observer.observe_event("EVENT", EventType.IP, time)
        assert result == "new_value"


class TestObserverWithStateAndAttributes:
    """Test observers with state and attributes parameters."""

    def test_observe_risk_factor_accepts_state_and_attributes(self):
        """observe_risk_factor accepts optional state and attributes."""
        observer = ConstantRiskFactorObserver(1.0)
        time = ActusDateTime(2024, 6, 15, 0, 0, 0)

        tmd = ActusDateTime(2029, 1, 15, 0, 0, 0)
        sd = ActusDateTime(2024, 1, 15, 0, 0, 0)
        state = ContractState(
            tmd=tmd,
            sd=sd,
            nt=jnp.array(100000.0),
            ipnr=jnp.array(0.05),
            ipac=jnp.array(0.0),
            feac=jnp.array(0.0),
            nsc=jnp.array(1.0),
            isc=jnp.array(1.0),
        )

        attributes = ContractAttributes(
            contract_id="TEST001",
            contract_type=ContractType.PAM,
            contract_role=ContractRole.RPA,
            status_date=sd,
            currency="USD",
        )

        # Should not raise any errors
        result = observer.observe_risk_factor("RATE", time, state=state, attributes=attributes)
        assert jnp.allclose(result, jnp.array(1.0))

    def test_observe_event_accepts_state_and_attributes(self):
        """observe_event accepts optional state and attributes."""
        observer = ConstantRiskFactorObserver(1.0)
        time = ActusDateTime(2024, 6, 15, 0, 0, 0)

        tmd = ActusDateTime(2029, 1, 15, 0, 0, 0)
        sd = ActusDateTime(2024, 1, 15, 0, 0, 0)
        state = ContractState(
            tmd=tmd,
            sd=sd,
            nt=jnp.array(100000.0),
            ipnr=jnp.array(0.05),
            ipac=jnp.array(0.0),
            feac=jnp.array(0.0),
            nsc=jnp.array(1.0),
            isc=jnp.array(1.0),
        )

        attributes = ContractAttributes(
            contract_id="TEST001",
            contract_type=ContractType.PAM,
            contract_role=ContractRole.RPA,
            status_date=sd,
            currency="USD",
        )

        # Should not raise any errors
        result = observer.observe_event(
            "EVENT", EventType.IP, time, state=state, attributes=attributes
        )
        assert jnp.allclose(result, jnp.array(1.0))


class TestJAXCompatibility:
    """Test JAX compatibility for observers."""

    def test_constant_observer_returns_jax_array(self):
        """ConstantRiskFactorObserver returns JAX arrays."""
        observer = ConstantRiskFactorObserver(1.5)
        time = ActusDateTime(2024, 6, 15, 0, 0, 0)

        result = observer.observe_risk_factor("RATE", time)

        assert isinstance(result, jnp.ndarray)
        assert result.dtype == jnp.float32

    def test_dict_observer_returns_jax_array(self):
        """DictRiskFactorObserver returns JAX arrays."""
        observer = DictRiskFactorObserver({"USD/EUR": 1.18})
        time = ActusDateTime(2024, 6, 15, 0, 0, 0)

        result = observer.observe_risk_factor("USD/EUR", time)

        assert isinstance(result, jnp.ndarray)
        assert result.dtype == jnp.float32

    def test_observer_values_can_be_used_in_jax_operations(self):
        """Observer values work in JAX operations."""
        observer = DictRiskFactorObserver({"RATE": 0.05, "NOTIONAL": 100000.0})
        time = ActusDateTime(2024, 6, 15, 0, 0, 0)

        rate = observer.observe_risk_factor("RATE", time)
        notional = observer.observe_risk_factor("NOTIONAL", time)

        # Perform JAX operations
        interest = notional * rate
        assert jnp.allclose(interest, jnp.array(5000.0))


class TestObserverEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_dict_observer(self):
        """DictRiskFactorObserver can be created with empty dict."""
        observer = DictRiskFactorObserver({})

        assert len(observer.risk_factors) == 0
        assert len(observer.event_data) == 0

    def test_zero_value_risk_factor(self):
        """Observer handles zero value risk factors."""
        observer = DictRiskFactorObserver({"ZERO_RATE": 0.0})
        time = ActusDateTime(2024, 6, 15, 0, 0, 0)

        result = observer.observe_risk_factor("ZERO_RATE", time)

        assert jnp.allclose(result, jnp.array(0.0, dtype=jnp.float32))

    def test_negative_value_risk_factor(self):
        """Observer handles negative value risk factors."""
        observer = DictRiskFactorObserver({"NEGATIVE_RATE": -0.01})
        time = ActusDateTime(2024, 6, 15, 0, 0, 0)

        result = observer.observe_risk_factor("NEGATIVE_RATE", time)

        assert jnp.allclose(result, jnp.array(-0.01, dtype=jnp.float32))

    def test_large_value_risk_factor(self):
        """Observer handles large value risk factors."""
        observer = DictRiskFactorObserver({"LARGE_VALUE": 1e9})
        time = ActusDateTime(2024, 6, 15, 0, 0, 0)

        result = observer.observe_risk_factor("LARGE_VALUE", time)

        assert jnp.allclose(result, jnp.array(1e9, dtype=jnp.float32))

    def test_observer_name_in_error_message(self):
        """Observer name appears in error messages."""
        observer = DictRiskFactorObserver({"RATE": 0.05}, name="MyCustomObserver")
        time = ActusDateTime(2024, 6, 15, 0, 0, 0)

        with pytest.raises(KeyError) as exc_info:
            observer.observe_risk_factor("MISSING", time)

        assert "MyCustomObserver" in str(exc_info.value)


class TestJaxRiskFactorObserver:
    """Test JaxRiskFactorObserver for JAX compatibility."""

    def test_init_with_array(self):
        """Initialize with JAX array."""
        risk_factors = jnp.array([1.18, 0.05, 100000.0])
        observer = JaxRiskFactorObserver(risk_factors)

        assert jnp.allclose(observer.risk_factors, risk_factors)
        assert observer.size == 3

    def test_init_with_default_value(self):
        """Initialize with custom default value."""
        risk_factors = jnp.array([1.0, 2.0])
        observer = JaxRiskFactorObserver(risk_factors, default_value=99.0)

        assert jnp.allclose(observer.default_value, jnp.array(99.0))

    def test_get_valid_index(self):
        """Get risk factor at valid index."""
        observer = JaxRiskFactorObserver(jnp.array([1.18, 0.05, 100000.0]))

        result0 = observer.get(0)
        result1 = observer.get(1)
        result2 = observer.get(2)

        assert jnp.allclose(result0, jnp.array(1.18))
        assert jnp.allclose(result1, jnp.array(0.05))
        assert jnp.allclose(result2, jnp.array(100000.0))

    def test_get_out_of_bounds_returns_default(self):
        """Out-of-bounds index returns default value."""
        observer = JaxRiskFactorObserver(jnp.array([1.0, 2.0]), default_value=0.0)

        result_neg = observer.get(-1)
        result_too_large = observer.get(10)

        assert jnp.allclose(result_neg, jnp.array(0.0))
        assert jnp.allclose(result_too_large, jnp.array(0.0))

    def test_get_batch(self):
        """Get multiple risk factors at once."""
        observer = JaxRiskFactorObserver(jnp.array([1.18, 0.05, 100000.0]))
        indices = jnp.array([0, 2])

        results = observer.get_batch(indices)

        expected = jnp.array([1.18, 100000.0])
        assert jnp.allclose(results, expected)

    def test_get_batch_with_out_of_bounds(self):
        """get_batch handles out-of-bounds indices."""
        observer = JaxRiskFactorObserver(jnp.array([1.0, 2.0, 3.0]), default_value=0.0)
        indices = jnp.array([0, 10, 1])

        results = observer.get_batch(indices)

        expected = jnp.array([1.0, 0.0, 2.0])  # Index 10 gets default
        assert jnp.allclose(results, expected)

    def test_update_creates_new_observer(self):
        """update returns new observer without modifying original."""
        observer = JaxRiskFactorObserver(jnp.array([1.18, 0.05]))

        new_observer = observer.update(0, 1.20)

        # Original unchanged
        assert jnp.allclose(observer.get(0), jnp.array(1.18))
        # New observer has updated value
        assert jnp.allclose(new_observer.get(0), jnp.array(1.20))
        # Other values unchanged
        assert jnp.allclose(new_observer.get(1), jnp.array(0.05))

    def test_update_batch_creates_new_observer(self):
        """update_batch returns new observer."""
        observer = JaxRiskFactorObserver(jnp.array([1.0, 2.0, 3.0]))

        new_observer = observer.update_batch(jnp.array([0, 2]), jnp.array([10.0, 30.0]))

        # Original unchanged
        assert jnp.allclose(observer.get(0), jnp.array(1.0))
        assert jnp.allclose(observer.get(2), jnp.array(3.0))
        # New observer has updated values
        assert jnp.allclose(new_observer.get(0), jnp.array(10.0))
        assert jnp.allclose(new_observer.get(1), jnp.array(2.0))
        assert jnp.allclose(new_observer.get(2), jnp.array(30.0))

    def test_to_array(self):
        """to_array returns the risk factors array."""
        risk_factors = jnp.array([1.18, 0.05, 100000.0])
        observer = JaxRiskFactorObserver(risk_factors)

        result = observer.to_array()

        assert jnp.allclose(result, risk_factors)

    def test_from_dict(self):
        """Create observer from dictionary."""
        observer = JaxRiskFactorObserver.from_dict({0: 1.18, 1: 0.05, 2: 100000.0})

        assert observer.size == 3
        assert jnp.allclose(observer.get(0), jnp.array(1.18))
        assert jnp.allclose(observer.get(1), jnp.array(0.05))
        assert jnp.allclose(observer.get(2), jnp.array(100000.0))

    def test_from_dict_with_size(self):
        """Create observer from dict with explicit size."""
        observer = JaxRiskFactorObserver.from_dict({0: 1.18, 2: 100000.0}, size=5)

        assert observer.size == 5
        assert jnp.allclose(observer.get(0), jnp.array(1.18))
        assert jnp.allclose(observer.get(1), jnp.array(0.0))  # Default
        assert jnp.allclose(observer.get(2), jnp.array(100000.0))
        assert jnp.allclose(observer.get(3), jnp.array(0.0))  # Default

    def test_from_dict_with_default_value(self):
        """Create observer from dict with custom default."""
        observer = JaxRiskFactorObserver.from_dict({0: 1.18}, size=3, default_value=99.0)

        assert jnp.allclose(observer.get(1), jnp.array(99.0))
        assert jnp.allclose(observer.get(2), jnp.array(99.0))

    def test_from_dict_empty(self):
        """Create observer from empty dict."""
        observer = JaxRiskFactorObserver.from_dict({})

        assert observer.size == 0


class TestJaxRiskFactorObserverJIT:
    """Test JIT compilation of JaxRiskFactorObserver."""

    def test_get_is_jit_compilable(self):
        """get method can be JIT-compiled."""
        observer = JaxRiskFactorObserver(jnp.array([1.18, 0.05, 100000.0]))

        @jax.jit
        def jitted_get(idx):
            return observer.get(idx)

        result = jitted_get(0)
        assert jnp.allclose(result, jnp.array(1.18))

    def test_get_batch_is_jit_compilable(self):
        """get_batch method can be JIT-compiled."""
        observer = JaxRiskFactorObserver(jnp.array([1.18, 0.05, 100000.0]))

        @jax.jit
        def jitted_get_batch(indices):
            return observer.get_batch(indices)

        indices = jnp.array([0, 2])
        result = jitted_get_batch(indices)
        expected = jnp.array([1.18, 100000.0])
        assert jnp.allclose(result, expected)

    def test_complex_jit_function(self):
        """Complex function using observer can be JIT-compiled."""
        risk_factors = jnp.array([1.18, 0.05, 100000.0])

        @jax.jit
        def contract_value(rf):
            observer = JaxRiskFactorObserver(rf)
            fx_rate = observer.get(0)
            rate = observer.get(1)
            notional = observer.get(2)
            return notional * rate * fx_rate

        result = contract_value(risk_factors)
        expected = 100000.0 * 0.05 * 1.18
        assert jnp.allclose(result, jnp.array(expected))


class TestJaxRiskFactorObserverGrad:
    """Test automatic differentiation of JaxRiskFactorObserver."""

    def test_grad_simple_function(self):
        """Compute gradient of simple function using observer."""
        risk_factors = jnp.array([2.0, 3.0])

        def f(rf):
            observer = JaxRiskFactorObserver(rf)
            x = observer.get(0)
            y = observer.get(1)
            return x * y  # f(x, y) = x * y

        # df/dx = y = 3.0, df/dy = x = 2.0
        grad_f = jax.grad(f)
        sensitivities = grad_f(risk_factors)

        expected = jnp.array([3.0, 2.0])
        assert jnp.allclose(sensitivities, expected)

    def test_grad_contract_value(self):
        """Compute sensitivities (Greeks) for contract value."""
        # Risk factors: [fx_rate, interest_rate, notional]
        risk_factors = jnp.array([1.18, 0.05, 100000.0])

        def contract_value(rf):
            observer = JaxRiskFactorObserver(rf)
            fx = observer.get(0)
            rate = observer.get(1)
            notional = observer.get(2)
            return notional * rate * fx

        # Compute sensitivities
        sensitivities = jax.grad(contract_value)(risk_factors)

        # d(value)/d(fx) = notional * rate = 100000 * 0.05 = 5000
        # d(value)/d(rate) = notional * fx = 100000 * 1.18 = 118000
        # d(value)/d(notional) = rate * fx = 0.05 * 1.18 = 0.059
        expected = jnp.array([5000.0, 118000.0, 0.059])
        assert jnp.allclose(sensitivities, expected, rtol=1e-5)

    def test_grad_with_get_batch(self):
        """Gradient works with get_batch."""
        risk_factors = jnp.array([2.0, 3.0, 5.0])

        def f(rf):
            observer = JaxRiskFactorObserver(rf)
            values = observer.get_batch(jnp.array([0, 2]))
            return values[0] * values[1]  # 2.0 * 5.0 = 10.0

        # df/drf[0] = 5.0, df/drf[1] = 0.0, df/drf[2] = 2.0
        sensitivities = jax.grad(f)(risk_factors)

        expected = jnp.array([5.0, 0.0, 2.0])
        assert jnp.allclose(sensitivities, expected)

    def test_second_order_derivative(self):
        """Compute second-order derivatives (Hessian diagonal)."""
        risk_factors = jnp.array([2.0, 3.0])

        def f(rf):
            observer = JaxRiskFactorObserver(rf)
            x = observer.get(0)
            y = observer.get(1)
            return x**2 * y  # f(x, y) = x^2 * y

        # df/dx = 2xy, d2f/dx2 = 2y
        # df/dy = x^2, d2f/dy2 = 0
        def grad_x(rf):
            return jax.grad(f)(rf)[0]

        def grad_y(rf):
            return jax.grad(f)(rf)[1]

        # Second derivatives
        d2f_dx2 = jax.grad(grad_x)(risk_factors)[0]
        d2f_dy2 = jax.grad(grad_y)(risk_factors)[1]

        # d2f/dx2 = 2y = 2*3 = 6.0
        # d2f/dy2 = 0
        assert jnp.allclose(d2f_dx2, jnp.array(6.0))
        assert jnp.allclose(d2f_dy2, jnp.array(0.0))


class TestJaxRiskFactorObserverVmap:
    """Test vectorization of JaxRiskFactorObserver with vmap."""

    def test_vmap_over_indices(self):
        """Vectorize get over multiple indices."""
        observer = JaxRiskFactorObserver(jnp.array([1.0, 2.0, 3.0, 4.0]))

        # Vectorize get over indices
        get_vectorized = jax.vmap(lambda idx: observer.get(idx))
        indices = jnp.array([0, 2, 1, 3])

        results = get_vectorized(indices)

        expected = jnp.array([1.0, 3.0, 2.0, 4.0])
        assert jnp.allclose(results, expected)

    def test_vmap_over_multiple_observers(self):
        """Vectorize computation over multiple risk factor scenarios."""
        # Multiple scenarios: different FX rates
        scenarios = jnp.array(
            [
                [1.10, 0.05, 100000.0],
                [1.18, 0.05, 100000.0],
                [1.25, 0.05, 100000.0],
            ]
        )

        def compute_value(rf):
            observer = JaxRiskFactorObserver(rf)
            fx = observer.get(0)
            rate = observer.get(1)
            notional = observer.get(2)
            return notional * rate * fx

        # Vectorize over scenarios
        values = jax.vmap(compute_value)(scenarios)

        expected = jnp.array([5500.0, 5900.0, 6250.0])
        assert jnp.allclose(values, expected)

    def test_vmap_with_grad(self):
        """Combine vmap and grad for batch sensitivity computation."""
        scenarios = jnp.array(
            [
                [1.0, 2.0],
                [2.0, 3.0],
                [3.0, 4.0],
            ]
        )

        def f(rf):
            observer = JaxRiskFactorObserver(rf)
            x = observer.get(0)
            y = observer.get(1)
            return x * y

        # Compute gradients for all scenarios
        sensitivities = jax.vmap(jax.grad(f))(scenarios)

        # For each scenario: df/dx = y, df/dy = x
        expected = jnp.array(
            [
                [2.0, 1.0],  # Scenario 1
                [3.0, 2.0],  # Scenario 2
                [4.0, 3.0],  # Scenario 3
            ]
        )
        assert jnp.allclose(sensitivities, expected)


class TestTimeSeriesRiskFactorObserver:
    """Test TimeSeriesRiskFactorObserver."""

    def test_init_sorts_series_by_time(self):
        """Series are sorted by time at construction."""
        # Provide unsorted input
        ts = {
            "RATE": [
                (ActusDateTime(2025, 1, 1), 0.05),
                (ActusDateTime(2024, 1, 1), 0.04),
                (ActusDateTime(2024, 7, 1), 0.045),
            ]
        }
        observer = TimeSeriesRiskFactorObserver(ts)
        # Step interpolation at a time between first two should return first
        result = observer.observe_risk_factor("RATE", ActusDateTime(2024, 4, 1))
        assert jnp.allclose(result, jnp.array(0.04, dtype=jnp.float32))

    def test_init_invalid_interpolation_raises(self):
        """Invalid interpolation method raises ValueError."""
        with pytest.raises(ValueError, match="interpolation"):
            TimeSeriesRiskFactorObserver(
                {"R": [(ActusDateTime(2024, 1, 1), 0.0)]}, interpolation="cubic"
            )

    def test_init_invalid_extrapolation_raises(self):
        """Invalid extrapolation method raises ValueError."""
        with pytest.raises(ValueError, match="extrapolation"):
            TimeSeriesRiskFactorObserver(
                {"R": [(ActusDateTime(2024, 1, 1), 0.0)]}, extrapolation="none"
            )

    def test_step_interpolation_at_exact_point(self):
        """Step interpolation returns exact value at data point."""
        ts = {
            "RATE": [
                (ActusDateTime(2024, 1, 1), 0.04),
                (ActusDateTime(2024, 7, 1), 0.045),
            ]
        }
        observer = TimeSeriesRiskFactorObserver(ts, interpolation="step")
        result = observer.observe_risk_factor("RATE", ActusDateTime(2024, 1, 1))
        assert jnp.allclose(result, jnp.array(0.04, dtype=jnp.float32))

    def test_step_interpolation_between_points(self):
        """Step interpolation returns left value between points."""
        ts = {
            "RATE": [
                (ActusDateTime(2024, 1, 1), 0.04),
                (ActusDateTime(2024, 7, 1), 0.05),
            ]
        }
        observer = TimeSeriesRiskFactorObserver(ts, interpolation="step")
        result = observer.observe_risk_factor("RATE", ActusDateTime(2024, 4, 1))
        assert jnp.allclose(result, jnp.array(0.04, dtype=jnp.float32))

    def test_linear_interpolation_at_exact_point(self):
        """Linear interpolation returns exact value at data point."""
        ts = {
            "RATE": [
                (ActusDateTime(2024, 1, 1), 0.04),
                (ActusDateTime(2024, 7, 1), 0.05),
            ]
        }
        observer = TimeSeriesRiskFactorObserver(ts, interpolation="linear")
        result = observer.observe_risk_factor("RATE", ActusDateTime(2024, 1, 1))
        assert jnp.allclose(result, jnp.array(0.04, dtype=jnp.float32))

    def test_linear_interpolation_midpoint(self):
        """Linear interpolation at midpoint returns average."""
        ts = {
            "RATE": [
                (ActusDateTime(2024, 1, 1), 0.04),
                (ActusDateTime(2025, 1, 1), 0.06),
            ]
        }
        observer = TimeSeriesRiskFactorObserver(ts, interpolation="linear")
        # Midpoint (approximately July 1-2 due to leap year)
        result = observer.observe_risk_factor("RATE", ActusDateTime(2024, 7, 1))
        # Should be close to 0.05 (proportional to days)
        assert 0.049 < float(result) < 0.051

    def test_flat_extrapolation_before_first(self):
        """Flat extrapolation returns first value before series start."""
        ts = {
            "RATE": [
                (ActusDateTime(2024, 1, 1), 0.04),
                (ActusDateTime(2025, 1, 1), 0.05),
            ]
        }
        observer = TimeSeriesRiskFactorObserver(ts, extrapolation="flat")
        result = observer.observe_risk_factor("RATE", ActusDateTime(2023, 1, 1))
        assert jnp.allclose(result, jnp.array(0.04, dtype=jnp.float32))

    def test_flat_extrapolation_after_last(self):
        """Flat extrapolation returns last value after series end."""
        ts = {
            "RATE": [
                (ActusDateTime(2024, 1, 1), 0.04),
                (ActusDateTime(2025, 1, 1), 0.05),
            ]
        }
        observer = TimeSeriesRiskFactorObserver(ts, extrapolation="flat")
        result = observer.observe_risk_factor("RATE", ActusDateTime(2026, 1, 1))
        assert jnp.allclose(result, jnp.array(0.05, dtype=jnp.float32))

    def test_raise_extrapolation_before_first(self):
        """Raise extrapolation raises KeyError before series start."""
        ts = {
            "RATE": [
                (ActusDateTime(2024, 1, 1), 0.04),
                (ActusDateTime(2025, 1, 1), 0.05),
            ]
        }
        observer = TimeSeriesRiskFactorObserver(ts, extrapolation="raise")
        with pytest.raises(KeyError, match="before first"):
            observer.observe_risk_factor("RATE", ActusDateTime(2023, 1, 1))

    def test_raise_extrapolation_after_last(self):
        """Raise extrapolation raises KeyError after series end."""
        ts = {
            "RATE": [
                (ActusDateTime(2024, 1, 1), 0.04),
                (ActusDateTime(2025, 1, 1), 0.05),
            ]
        }
        observer = TimeSeriesRiskFactorObserver(ts, extrapolation="raise")
        with pytest.raises(KeyError, match="after last"):
            observer.observe_risk_factor("RATE", ActusDateTime(2026, 1, 1))

    def test_unknown_identifier_raises_keyerror(self):
        """Unknown identifier raises KeyError."""
        ts = {"RATE": [(ActusDateTime(2024, 1, 1), 0.04)]}
        observer = TimeSeriesRiskFactorObserver(ts)
        with pytest.raises(KeyError, match="UNKNOWN"):
            observer.observe_risk_factor("UNKNOWN", ActusDateTime(2024, 1, 1))

    def test_event_data_step_interpolation(self):
        """Event data supports step interpolation."""
        event_data = {
            "PP_AMOUNT": [
                (ActusDateTime(2024, 1, 1), 1000.0),
                (ActusDateTime(2024, 7, 1), 2000.0),
            ]
        }
        observer = TimeSeriesRiskFactorObserver(
            risk_factors={},
            event_data=event_data,
        )
        result = observer.observe_event("PP_AMOUNT", EventType.PP, ActusDateTime(2024, 4, 1))
        assert result == 1000.0

    def test_event_data_unknown_raises_keyerror(self):
        """Unknown event data identifier raises KeyError."""
        observer = TimeSeriesRiskFactorObserver(risk_factors={})
        with pytest.raises(KeyError, match="UNKNOWN"):
            observer.observe_event("UNKNOWN", EventType.PP, ActusDateTime(2024, 1, 1))

    def test_is_risk_factor_observer(self):
        """TimeSeriesRiskFactorObserver implements RiskFactorObserver protocol."""
        ts = {"RATE": [(ActusDateTime(2024, 1, 1), 0.04)]}
        observer = TimeSeriesRiskFactorObserver(ts)
        assert isinstance(observer, RiskFactorObserver)

    def test_returns_jax_array(self):
        """Returns JAX array with float32 dtype."""
        ts = {"RATE": [(ActusDateTime(2024, 1, 1), 0.04)]}
        observer = TimeSeriesRiskFactorObserver(ts)
        result = observer.observe_risk_factor("RATE", ActusDateTime(2024, 1, 1))
        assert isinstance(result, jnp.ndarray)
        assert result.dtype == jnp.float32

    def test_single_point_series(self):
        """Single-point series works for any query time with flat extrapolation."""
        ts = {"RATE": [(ActusDateTime(2024, 6, 1), 0.05)]}
        observer = TimeSeriesRiskFactorObserver(ts, extrapolation="flat")
        assert jnp.allclose(
            observer.observe_risk_factor("RATE", ActusDateTime(2024, 1, 1)),
            jnp.array(0.05, dtype=jnp.float32),
        )
        assert jnp.allclose(
            observer.observe_risk_factor("RATE", ActusDateTime(2024, 6, 1)),
            jnp.array(0.05, dtype=jnp.float32),
        )


class TestCurveRiskFactorObserver:
    """Test CurveRiskFactorObserver."""

    def test_init_sorts_curve_by_tenor(self):
        """Curve is sorted by tenor at construction."""
        curve = {
            "YIELD": [
                (5.0, 0.05),
                (0.25, 0.03),
                (1.0, 0.04),
            ]
        }
        observer = CurveRiskFactorObserver(
            curves=curve,
            reference_date=ActusDateTime(2024, 1, 1),
        )
        # At ~3 months (0.25y), should get 0.03
        result = observer.observe_risk_factor("YIELD", ActusDateTime(2024, 4, 1))
        assert 0.029 < float(result) < 0.041

    def test_init_invalid_interpolation_raises(self):
        """Invalid interpolation raises ValueError."""
        with pytest.raises(ValueError, match="interpolation"):
            CurveRiskFactorObserver(curves={"Y": [(1.0, 0.04)]}, interpolation="cubic")

    def test_linear_interpolation_at_exact_tenor(self):
        """Linear interpolation returns exact rate at tenor point."""
        curve = {"YIELD": [(1.0, 0.04), (2.0, 0.05)]}
        observer = CurveRiskFactorObserver(
            curves=curve,
            reference_date=ActusDateTime(2024, 1, 1),
        )
        # At exactly 1 year
        result = observer.observe_risk_factor("YIELD", ActusDateTime(2025, 1, 1))
        assert jnp.allclose(result, jnp.array(0.04, dtype=jnp.float32), atol=1e-3)

    def test_linear_interpolation_between_tenors(self):
        """Linear interpolation between tenor points."""
        curve = {"YIELD": [(1.0, 0.04), (3.0, 0.06)]}
        observer = CurveRiskFactorObserver(
            curves=curve,
            reference_date=ActusDateTime(2024, 1, 1),
        )
        # At 2 years (midpoint of 1y-3y), should be ~0.05
        result = observer.observe_risk_factor("YIELD", ActusDateTime(2026, 1, 1))
        assert 0.049 < float(result) < 0.051

    def test_flat_extrapolation_before_first_tenor(self):
        """Flat extrapolation returns first rate before curve start."""
        curve = {"YIELD": [(1.0, 0.04), (5.0, 0.06)]}
        observer = CurveRiskFactorObserver(
            curves=curve,
            reference_date=ActusDateTime(2024, 1, 1),
        )
        # At 0.1 years (before first tenor)
        result = observer.observe_risk_factor("YIELD", ActusDateTime(2024, 2, 1))
        assert jnp.allclose(result, jnp.array(0.04, dtype=jnp.float32))

    def test_flat_extrapolation_after_last_tenor(self):
        """Flat extrapolation returns last rate after curve end."""
        curve = {"YIELD": [(1.0, 0.04), (5.0, 0.06)]}
        observer = CurveRiskFactorObserver(
            curves=curve,
            reference_date=ActusDateTime(2024, 1, 1),
        )
        # At 10 years
        result = observer.observe_risk_factor("YIELD", ActusDateTime(2034, 1, 1))
        assert jnp.allclose(result, jnp.array(0.06, dtype=jnp.float32))

    def test_log_linear_interpolation(self):
        """Log-linear interpolation computes correctly."""
        curve = {"YIELD": [(1.0, 0.04), (3.0, 0.06)]}
        observer = CurveRiskFactorObserver(
            curves=curve,
            reference_date=ActusDateTime(2024, 1, 1),
            interpolation="log_linear",
        )
        # At 2 years (midpoint)
        result = observer.observe_risk_factor("YIELD", ActusDateTime(2026, 1, 1))
        # Log-linear midpoint: exp(avg(log(0.04), log(0.06))) ~= 0.04899
        assert 0.048 < float(result) < 0.051

    def test_log_linear_rejects_non_positive_rates(self):
        """Log-linear interpolation rejects non-positive rates."""
        with pytest.raises(ValueError, match="positive rates"):
            CurveRiskFactorObserver(
                curves={"YIELD": [(1.0, 0.0), (2.0, 0.05)]},
                interpolation="log_linear",
            )

    def test_reference_date_from_attributes(self):
        """Falls back to attributes.status_date when no reference_date."""
        curve = {"YIELD": [(1.0, 0.04), (5.0, 0.06)]}
        observer = CurveRiskFactorObserver(curves=curve)
        attrs = ContractAttributes(
            contract_id="TEST001",
            contract_type=ContractType.PAM,
            contract_role=ContractRole.RPA,
            status_date=ActusDateTime(2024, 1, 1),
        )
        result = observer.observe_risk_factor("YIELD", ActusDateTime(2025, 1, 1), attributes=attrs)
        assert jnp.allclose(result, jnp.array(0.04, dtype=jnp.float32), atol=1e-3)

    def test_no_reference_date_no_attributes_raises(self):
        """Raises ValueError when no reference date available."""
        curve = {"YIELD": [(1.0, 0.04)]}
        observer = CurveRiskFactorObserver(curves=curve)
        with pytest.raises(ValueError, match="reference_date"):
            observer.observe_risk_factor("YIELD", ActusDateTime(2025, 1, 1))

    def test_unknown_identifier_raises_keyerror(self):
        """Unknown identifier raises KeyError."""
        observer = CurveRiskFactorObserver(
            curves={"YIELD": [(1.0, 0.04)]},
            reference_date=ActusDateTime(2024, 1, 1),
        )
        with pytest.raises(KeyError, match="UNKNOWN"):
            observer.observe_risk_factor("UNKNOWN", ActusDateTime(2025, 1, 1))

    def test_event_data_raises_keyerror(self):
        """Event data raises KeyError (not supported)."""
        observer = CurveRiskFactorObserver(
            curves={"YIELD": [(1.0, 0.04)]},
            reference_date=ActusDateTime(2024, 1, 1),
        )
        with pytest.raises(KeyError, match="does not support"):
            observer.observe_event("YIELD", EventType.RR, ActusDateTime(2025, 1, 1))

    def test_is_risk_factor_observer(self):
        """CurveRiskFactorObserver implements RiskFactorObserver protocol."""
        observer = CurveRiskFactorObserver(
            curves={"YIELD": [(1.0, 0.04)]},
            reference_date=ActusDateTime(2024, 1, 1),
        )
        assert isinstance(observer, RiskFactorObserver)

    def test_returns_jax_array(self):
        """Returns JAX array with float32 dtype."""
        observer = CurveRiskFactorObserver(
            curves={"YIELD": [(1.0, 0.04)]},
            reference_date=ActusDateTime(2024, 1, 1),
        )
        result = observer.observe_risk_factor("YIELD", ActusDateTime(2025, 1, 1))
        assert isinstance(result, jnp.ndarray)
        assert result.dtype == jnp.float32


class TestCallbackRiskFactorObserver:
    """Test CallbackRiskFactorObserver."""

    def test_callback_receives_identifier_and_time(self):
        """Callback receives the correct identifier and time."""
        calls = []

        def my_callback(identifier, time):
            calls.append((identifier, time))
            return 0.05

        observer = CallbackRiskFactorObserver(callback=my_callback)
        t = ActusDateTime(2024, 6, 15)
        observer.observe_risk_factor("LIBOR", t)

        assert len(calls) == 1
        assert calls[0] == ("LIBOR", t)

    def test_callback_result_wrapped_as_jax_array(self):
        """Callback result is wrapped as JAX array."""
        observer = CallbackRiskFactorObserver(callback=lambda i, t: 0.05)
        result = observer.observe_risk_factor("RATE", ActusDateTime(2024, 1, 1))
        assert isinstance(result, jnp.ndarray)
        assert result.dtype == jnp.float32
        assert jnp.allclose(result, jnp.array(0.05, dtype=jnp.float32))

    def test_event_callback_receives_identifier_type_time(self):
        """Event callback receives identifier, event_type, and time."""
        calls = []

        def my_event_callback(identifier, event_type, time):
            calls.append((identifier, event_type, time))
            return 5000.0

        observer = CallbackRiskFactorObserver(
            callback=lambda i, t: 0.0,
            event_callback=my_event_callback,
        )
        t = ActusDateTime(2024, 6, 15)
        result = observer.observe_event("PP", EventType.PP, t)

        assert len(calls) == 1
        assert calls[0] == ("PP", EventType.PP, t)
        assert result == 5000.0

    def test_no_event_callback_raises_keyerror(self):
        """Missing event callback raises KeyError."""
        observer = CallbackRiskFactorObserver(callback=lambda i, t: 0.0)
        with pytest.raises(KeyError, match="No event callback"):
            observer.observe_event("PP", EventType.PP, ActusDateTime(2024, 1, 1))

    def test_callback_exception_propagates(self):
        """Exceptions from callback propagate."""

        def bad_callback(identifier, time):
            raise ValueError("bad data")

        observer = CallbackRiskFactorObserver(callback=bad_callback)
        with pytest.raises(ValueError, match="bad data"):
            observer.observe_risk_factor("RATE", ActusDateTime(2024, 1, 1))

    def test_is_risk_factor_observer(self):
        """CallbackRiskFactorObserver implements RiskFactorObserver protocol."""
        observer = CallbackRiskFactorObserver(callback=lambda i, t: 0.0)
        assert isinstance(observer, RiskFactorObserver)


class TestCompositeRiskFactorObserver:
    """Test CompositeRiskFactorObserver."""

    def test_init_empty_observers_raises(self):
        """Empty observers list raises ValueError."""
        with pytest.raises(ValueError, match="must not be empty"):
            CompositeRiskFactorObserver([])

    def test_first_observer_success(self):
        """Returns result from first observer when it succeeds."""
        obs1 = DictRiskFactorObserver({"RATE": 0.04})
        obs2 = DictRiskFactorObserver({"RATE": 0.05})
        composite = CompositeRiskFactorObserver([obs1, obs2])

        result = composite.observe_risk_factor("RATE", ActusDateTime(2024, 1, 1))
        assert jnp.allclose(result, jnp.array(0.04, dtype=jnp.float32))

    def test_fallback_to_second_observer(self):
        """Falls back to second observer when first raises KeyError."""
        obs1 = DictRiskFactorObserver({"FX": 1.18})
        obs2 = DictRiskFactorObserver({"RATE": 0.05})
        composite = CompositeRiskFactorObserver([obs1, obs2])

        result = composite.observe_risk_factor("RATE", ActusDateTime(2024, 1, 1))
        assert jnp.allclose(result, jnp.array(0.05, dtype=jnp.float32))

    def test_all_observers_fail_raises_keyerror(self):
        """Raises KeyError when all observers fail."""
        obs1 = DictRiskFactorObserver({"FX": 1.18})
        obs2 = DictRiskFactorObserver({"RATE": 0.05})
        composite = CompositeRiskFactorObserver([obs1, obs2])

        with pytest.raises(KeyError, match="not found in any observer"):
            composite.observe_risk_factor("UNKNOWN", ActusDateTime(2024, 1, 1))

    def test_non_keyerror_propagates_immediately(self):
        """Non-KeyError exceptions propagate without trying next observer."""

        def bad_callback(identifier, time):
            raise ValueError("bad data")

        obs1 = CallbackRiskFactorObserver(callback=bad_callback)
        obs2 = ConstantRiskFactorObserver(0.0)
        composite = CompositeRiskFactorObserver([obs1, obs2])

        with pytest.raises(ValueError, match="bad data"):
            composite.observe_risk_factor("RATE", ActusDateTime(2024, 1, 1))

    def test_event_data_fallback(self):
        """Event data falls back through observer chain."""
        obs1 = DictRiskFactorObserver({"R": 0.0}, event_data={"A": 100.0})
        obs2 = DictRiskFactorObserver({"R": 0.0}, event_data={"B": 200.0})
        composite = CompositeRiskFactorObserver([obs1, obs2])

        result = composite.observe_event("B", EventType.PP, ActusDateTime(2024, 1, 1))
        assert result == 200.0

    def test_timeseries_with_constant_fallback(self):
        """TimeSeriesRiskFactorObserver with ConstantRiskFactorObserver fallback."""
        ts = TimeSeriesRiskFactorObserver(
            {
                "LIBOR-3M": [
                    (ActusDateTime(2024, 1, 1), 0.04),
                    (ActusDateTime(2025, 1, 1), 0.05),
                ]
            }
        )
        fallback = ConstantRiskFactorObserver(0.0)
        composite = CompositeRiskFactorObserver([ts, fallback])

        # Known identifier uses time series
        result = composite.observe_risk_factor("LIBOR-3M", ActusDateTime(2024, 6, 1))
        assert jnp.allclose(result, jnp.array(0.04, dtype=jnp.float32))

        # Unknown identifier falls back to constant
        result = composite.observe_risk_factor("OTHER", ActusDateTime(2024, 6, 1))
        assert jnp.allclose(result, jnp.array(0.0, dtype=jnp.float32))

    def test_is_risk_factor_observer(self):
        """CompositeRiskFactorObserver implements RiskFactorObserver protocol."""
        composite = CompositeRiskFactorObserver([ConstantRiskFactorObserver(0.0)])
        assert isinstance(composite, RiskFactorObserver)
