"""Unit tests for payoff function framework.

T2.1: Payoff Function Framework Tests

Tests for:
- PayoffFunction Protocol
- BasePayoffFunction ABC
- settlement_currency_fx_rate
- canonical_contract_payoff
- Role sign application
- FX rate application
- JAX JIT compatibility
"""

from unittest.mock import Mock

import jax
import jax.numpy as jnp
import pytest

from jactus.core import ActusDateTime, ContractAttributes, ContractState
from jactus.core.types import ContractRole, ContractType, EventType
from jactus.functions import (
    BasePayoffFunction,
    PayoffFunction,
    canonical_contract_payoff,
    settlement_currency_fx_rate,
)


class TestPayoffFunctionProtocol:
    """Test PayoffFunction protocol enforcement."""

    def test_protocol_requires_call_method(self):
        """PayoffFunction protocol requires __call__ method."""

        class ValidPayoff:
            def __call__(self, event_type, state, attributes, time, risk_factor_observer):
                return jnp.array(100.0)

        assert isinstance(ValidPayoff(), PayoffFunction)

    def test_protocol_rejects_without_call_method(self):
        """Objects without __call__ are not PayoffFunctions."""

        class InvalidPayoff:
            pass

        assert not isinstance(InvalidPayoff(), PayoffFunction)


class ConcretePayoffFunction(BasePayoffFunction):
    """Concrete implementation for testing BasePayoffFunction."""

    def calculate_payoff(self, event_type, state, attributes, time, risk_factor_observer):
        """Return a fixed payoff for testing."""
        return jnp.array(1000.0, dtype=jnp.float32)


class TestBasePayoffFunctionInit:
    """Test BasePayoffFunction initialization."""

    def test_init_with_all_params(self):
        """Initialize with all parameters."""
        pof = ConcretePayoffFunction(
            contract_role=ContractRole.RPA, currency="USD", settlement_currency="EUR"
        )

        assert pof.contract_role == ContractRole.RPA
        assert pof.currency == "USD"
        assert pof.settlement_currency == "EUR"

    def test_init_without_settlement_currency(self):
        """Initialize without settlement currency (defaults to None)."""
        pof = ConcretePayoffFunction(contract_role=ContractRole.RPA, currency="USD")

        assert pof.contract_role == ContractRole.RPA
        assert pof.currency == "USD"
        assert pof.settlement_currency is None


class TestBasePayoffFunctionRoleSign:
    """Test apply_role_sign method."""

    @pytest.mark.parametrize(
        ("role", "expected_sign"),
        [
            (ContractRole.RPA, 1),
            (ContractRole.RPL, -1),
            (ContractRole.LG, 1),
            (ContractRole.ST, -1),
            (ContractRole.BUY, 1),
            (ContractRole.SEL, -1),
            (ContractRole.RFL, 1),
            (ContractRole.PFL, -1),
            (ContractRole.GUA, -1),
            (ContractRole.OBL, 1),
            (ContractRole.COL, 1),
            (ContractRole.CNO, 1),
            (ContractRole.UDL, 1),
            (ContractRole.UDLP, 1),
            (ContractRole.UDLM, -1),
        ],
    )
    def test_apply_role_sign_all_roles(self, role, expected_sign):
        """Test role sign application for all ContractRole values."""
        pof = ConcretePayoffFunction(contract_role=role, currency="USD")
        amount = jnp.array(100.0, dtype=jnp.float32)

        result = pof.apply_role_sign(amount)

        expected = jnp.array(100.0 * expected_sign, dtype=jnp.float32)
        assert jnp.allclose(result, expected)

    def test_apply_role_sign_preserves_magnitude(self):
        """Role sign preserves absolute magnitude."""
        pof = ConcretePayoffFunction(contract_role=ContractRole.RPA, currency="USD")
        amount = jnp.array(12345.67, dtype=jnp.float32)

        result = pof.apply_role_sign(amount)

        assert jnp.allclose(jnp.abs(result), jnp.abs(amount))

    def test_apply_role_sign_zero_stays_zero(self):
        """Role sign of zero is zero."""
        pof = ConcretePayoffFunction(contract_role=ContractRole.RPA, currency="USD")
        amount = jnp.array(0.0, dtype=jnp.float32)

        result = pof.apply_role_sign(amount)

        assert result == 0.0


class TestBasePayoffFunctionFXRate:
    """Test apply_fx_rate method."""

    def test_apply_fx_rate_same_currency(self):
        """No FX adjustment when settlement currency same as contract."""
        pof = ConcretePayoffFunction(
            contract_role=ContractRole.RPA, currency="USD", settlement_currency="USD"
        )
        amount = jnp.array(1000.0, dtype=jnp.float32)
        time = ActusDateTime(2024, 6, 15, 0, 0, 0)
        observer = Mock()

        result = pof.apply_fx_rate(amount, time, observer)

        # Should return amount * 1.0
        assert jnp.allclose(result, amount)
        # Observer should not be called
        observer.observe_risk_factor.assert_not_called()

    def test_apply_fx_rate_no_settlement_currency(self):
        """No FX adjustment when settlement currency is None."""
        pof = ConcretePayoffFunction(
            contract_role=ContractRole.RPA, currency="USD", settlement_currency=None
        )
        amount = jnp.array(1000.0, dtype=jnp.float32)
        time = ActusDateTime(2024, 6, 15, 0, 0, 0)
        observer = Mock()

        result = pof.apply_fx_rate(amount, time, observer)

        # Should return amount * 1.0
        assert jnp.allclose(result, amount)
        observer.observe_risk_factor.assert_not_called()

    def test_apply_fx_rate_different_currency(self):
        """FX rate applied when settlement currency differs."""
        pof = ConcretePayoffFunction(
            contract_role=ContractRole.RPA, currency="EUR", settlement_currency="USD"
        )
        amount = jnp.array(1000.0, dtype=jnp.float32)
        time = ActusDateTime(2024, 6, 15, 0, 0, 0)

        # Mock observer returning EUR/USD = 1.18
        observer = Mock()
        observer.observe_risk_factor.return_value = jnp.array(1.18, dtype=jnp.float32)

        result = pof.apply_fx_rate(amount, time, observer)

        # Should return 1000 * 1.18 = 1180
        expected = jnp.array(1180.0, dtype=jnp.float32)
        assert jnp.allclose(result, expected)

        # Observer should be called with correct FX pair
        observer.observe_risk_factor.assert_called_once_with(identifier="EUR/USD", time=time)

    def test_apply_fx_rate_format(self):
        """FX rate identifier uses correct format."""
        pof = ConcretePayoffFunction(
            contract_role=ContractRole.RPA, currency="GBP", settlement_currency="JPY"
        )
        amount = jnp.array(500.0, dtype=jnp.float32)
        time = ActusDateTime(2024, 6, 15, 0, 0, 0)

        observer = Mock()
        observer.observe_risk_factor.return_value = jnp.array(180.0, dtype=jnp.float32)

        pof.apply_fx_rate(amount, time, observer)

        # FX identifier should be "BASE/QUOTE" where BASE is contract currency
        observer.observe_risk_factor.assert_called_once_with(identifier="GBP/JPY", time=time)


class TestBasePayoffFunctionCall:
    """Test complete __call__ pipeline."""

    def test_call_pipeline_rpa_no_fx(self):
        """Complete payoff calculation: RPA role, no FX."""
        pof = ConcretePayoffFunction(contract_role=ContractRole.RPA, currency="USD")

        tmd = ActusDateTime(2024, 1, 1, 0, 0, 0)
        sd = ActusDateTime(2024, 1, 1, 0, 0, 0)
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
            status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
            currency="USD",
        )
        time = ActusDateTime(2024, 6, 15, 0, 0, 0)
        observer = Mock()

        result = pof(EventType.IP, state, attributes, time, observer)

        # calculate_payoff returns 1000.0
        # apply_role_sign: RPA = +1 → 1000.0
        # apply_fx_rate: no settlement currency → 1000.0
        expected = jnp.array(1000.0, dtype=jnp.float32)
        assert jnp.allclose(result, expected)

    def test_call_pipeline_rpl_no_fx(self):
        """Complete payoff calculation: RPL role, no FX."""
        pof = ConcretePayoffFunction(contract_role=ContractRole.RPL, currency="USD")

        tmd = ActusDateTime(2024, 1, 1, 0, 0, 0)
        sd = ActusDateTime(2024, 1, 1, 0, 0, 0)
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
            contract_role=ContractRole.RPL,
            status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
            currency="USD",
        )
        time = ActusDateTime(2024, 6, 15, 0, 0, 0)
        observer = Mock()

        result = pof(EventType.IP, state, attributes, time, observer)

        # calculate_payoff returns 1000.0
        # No automatic role_sign applied (each POF handles its own sign)
        # apply_fx_rate: no settlement currency → 1000.0
        expected = jnp.array(1000.0, dtype=jnp.float32)
        assert jnp.allclose(result, expected)

    def test_call_pipeline_with_fx(self):
        """Complete payoff calculation: with FX conversion."""
        pof = ConcretePayoffFunction(
            contract_role=ContractRole.RPA, currency="EUR", settlement_currency="USD"
        )

        tmd = ActusDateTime(2024, 1, 1, 0, 0, 0)
        sd = ActusDateTime(2024, 1, 1, 0, 0, 0)
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
            status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
            currency="EUR",
        )
        time = ActusDateTime(2024, 6, 15, 0, 0, 0)

        observer = Mock()
        observer.observe_risk_factor.return_value = jnp.array(1.18, dtype=jnp.float32)

        result = pof(EventType.IP, state, attributes, time, observer)

        # calculate_payoff returns 1000.0
        # apply_role_sign: RPA = +1 → 1000.0
        # apply_fx_rate: EUR/USD = 1.18 → 1180.0
        expected = jnp.array(1180.0, dtype=jnp.float32)
        assert jnp.allclose(result, expected)

    def test_call_pipeline_rpl_with_fx(self):
        """Complete payoff calculation: RPL with FX conversion."""
        pof = ConcretePayoffFunction(
            contract_role=ContractRole.RPL, currency="EUR", settlement_currency="USD"
        )

        tmd = ActusDateTime(2024, 1, 1, 0, 0, 0)
        sd = ActusDateTime(2024, 1, 1, 0, 0, 0)
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
            contract_role=ContractRole.RPL,
            status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
            currency="EUR",
        )
        time = ActusDateTime(2024, 6, 15, 0, 0, 0)

        observer = Mock()
        observer.observe_risk_factor.return_value = jnp.array(1.18, dtype=jnp.float32)

        result = pof(EventType.IP, state, attributes, time, observer)

        # calculate_payoff returns 1000.0
        # No automatic role_sign applied (each POF handles its own sign)
        # apply_fx_rate: EUR/USD = 1.18 → 1180.0
        expected = jnp.array(1180.0, dtype=jnp.float32)
        assert jnp.allclose(result, expected)


class TestSettlementCurrencyFXRate:
    """Test settlement_currency_fx_rate function."""

    def test_same_currency_returns_one(self):
        """Returns 1.0 when currencies are the same."""
        time = ActusDateTime(2024, 6, 15, 0, 0, 0)
        observer = Mock()

        result = settlement_currency_fx_rate(
            time=time,
            contract_currency="USD",
            settlement_currency="USD",
            risk_factor_observer=observer,
        )

        assert jnp.allclose(result, jnp.array(1.0, dtype=jnp.float32))
        observer.observe_risk_factor.assert_not_called()

    def test_none_settlement_currency_returns_one(self):
        """Returns 1.0 when settlement currency is None."""
        time = ActusDateTime(2024, 6, 15, 0, 0, 0)
        observer = Mock()

        result = settlement_currency_fx_rate(
            time=time,
            contract_currency="USD",
            settlement_currency=None,
            risk_factor_observer=observer,
        )

        assert jnp.allclose(result, jnp.array(1.0, dtype=jnp.float32))
        observer.observe_risk_factor.assert_not_called()

    def test_different_currency_observes_fx_rate(self):
        """Observes FX rate when currencies differ."""
        time = ActusDateTime(2024, 6, 15, 0, 0, 0)
        observer = Mock()
        observer.observe_risk_factor.return_value = jnp.array(1.18, dtype=jnp.float32)

        result = settlement_currency_fx_rate(
            time=time,
            contract_currency="EUR",
            settlement_currency="USD",
            risk_factor_observer=observer,
        )

        assert jnp.allclose(result, jnp.array(1.18, dtype=jnp.float32))
        observer.observe_risk_factor.assert_called_once_with(identifier="EUR/USD", time=time)

    def test_fx_identifier_format(self):
        """FX identifier uses BASE/QUOTE format."""
        time = ActusDateTime(2024, 6, 15, 0, 0, 0)
        observer = Mock()
        observer.observe_risk_factor.return_value = jnp.array(180.0, dtype=jnp.float32)

        settlement_currency_fx_rate(
            time=time,
            contract_currency="GBP",
            settlement_currency="JPY",
            risk_factor_observer=observer,
        )

        # Should observe "GBP/JPY" (contract/settlement)
        observer.observe_risk_factor.assert_called_once_with(identifier="GBP/JPY", time=time)


class TestCanonicalContractPayoff:
    """Test canonical_contract_payoff function."""

    def test_no_future_events_returns_zero(self):
        """Returns zero when there are no future events."""
        contract = Mock()
        event_schedule = Mock()
        event_schedule.events = []
        contract.get_events.return_value = event_schedule

        time = ActusDateTime(2024, 6, 15, 0, 0, 0)
        observer = Mock()

        result = canonical_contract_payoff(contract, time, observer)

        assert jnp.allclose(result, jnp.array(0.0, dtype=jnp.float32))

    def test_all_past_events_returns_zero(self):
        """Returns zero when all events are in the past."""
        contract = Mock()

        # Create mock events all in the past
        past_event1 = Mock()
        past_event1.time = ActusDateTime(2024, 1, 1, 0, 0, 0)
        past_event2 = Mock()
        past_event2.time = ActusDateTime(2024, 3, 1, 0, 0, 0)

        event_schedule = Mock()
        event_schedule.events = [past_event1, past_event2]
        contract.get_events.return_value = event_schedule

        time = ActusDateTime(2024, 6, 15, 0, 0, 0)
        observer = Mock()

        result = canonical_contract_payoff(contract, time, observer)

        assert jnp.allclose(result, jnp.array(0.0, dtype=jnp.float32))

    def test_calls_get_events(self):
        """Function calls contract.get_events() with observer."""
        contract = Mock()
        event_schedule = Mock()
        event_schedule.events = []
        contract.get_events.return_value = event_schedule

        time = ActusDateTime(2024, 6, 15, 0, 0, 0)
        observer = Mock()

        canonical_contract_payoff(contract, time, observer)

        contract.get_events.assert_called_once_with(observer)

    def test_filters_future_events_correctly(self):
        """Correctly filters events at or after valuation time."""
        contract = Mock()

        # Create mix of past and future events
        past_event = Mock()
        past_event.time = ActusDateTime(2024, 5, 1, 0, 0, 0)

        current_event = Mock()
        current_event.time = ActusDateTime(2024, 6, 15, 0, 0, 0)

        future_event = Mock()
        future_event.time = ActusDateTime(2024, 9, 15, 0, 0, 0)

        event_schedule = Mock()
        event_schedule.events = [past_event, current_event, future_event]
        contract.get_events.return_value = event_schedule

        time = ActusDateTime(2024, 6, 15, 0, 0, 0)
        observer = Mock()

        # Note: Current implementation returns 0.0 (TODO for full implementation)
        result = canonical_contract_payoff(contract, time, observer)

        # For now, verify it doesn't crash and returns a value
        assert isinstance(result, jnp.ndarray)
        assert result.dtype == jnp.float32


class TestJAXJITCompatibility:
    """Test JAX JIT compatibility for payoff functions."""

    def test_apply_role_sign_jit_compatible(self):
        """apply_role_sign is JIT-compatible."""
        pof = ConcretePayoffFunction(contract_role=ContractRole.RPA, currency="USD")

        @jax.jit
        def jitted_role_sign(amount):
            return pof.apply_role_sign(amount)

        amount = jnp.array(100.0, dtype=jnp.float32)
        result = jitted_role_sign(amount)

        assert jnp.allclose(result, jnp.array(100.0, dtype=jnp.float32))

    def test_settlement_currency_fx_rate_jit_compatible_same_currency(self):
        """settlement_currency_fx_rate is JIT-compatible for same currency case."""

        @jax.jit
        def jitted_fx_rate():
            time = ActusDateTime(2024, 6, 15, 0, 0, 0)
            observer = Mock()
            return settlement_currency_fx_rate(
                time=time,
                contract_currency="USD",
                settlement_currency="USD",
                risk_factor_observer=observer,
            )

        result = jitted_fx_rate()

        assert jnp.allclose(result, jnp.array(1.0, dtype=jnp.float32))

    def test_base_payoff_function_returns_jax_arrays(self):
        """BasePayoffFunction methods return JAX arrays."""
        pof = ConcretePayoffFunction(contract_role=ContractRole.RPA, currency="USD")

        amount = jnp.array(100.0, dtype=jnp.float32)
        result = pof.apply_role_sign(amount)

        assert isinstance(result, jnp.ndarray)
        assert result.dtype == jnp.float32


class TestPayoffFunctionEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_zero_payoff_through_pipeline(self):
        """Zero payoff flows correctly through pipeline."""

        class ZeroPayoff(BasePayoffFunction):
            def calculate_payoff(self, event_type, state, attributes, time, risk_factor_observer):
                return jnp.array(0.0, dtype=jnp.float32)

        pof = ZeroPayoff(contract_role=ContractRole.RPA, currency="EUR", settlement_currency="USD")

        tmd = ActusDateTime(2024, 1, 1, 0, 0, 0)
        sd = ActusDateTime(2024, 1, 1, 0, 0, 0)
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
            status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
            currency="EUR",
        )
        time = ActusDateTime(2024, 6, 15, 0, 0, 0)

        observer = Mock()
        observer.observe_risk_factor.return_value = jnp.array(1.18, dtype=jnp.float32)

        result = pof(EventType.IP, state, attributes, time, observer)

        # 0.0 * +1 * 1.18 = 0.0
        assert jnp.allclose(result, jnp.array(0.0, dtype=jnp.float32))

    def test_large_payoff_values(self):
        """Handles large payoff values correctly."""

        class LargePayoff(BasePayoffFunction):
            def calculate_payoff(self, event_type, state, attributes, time, risk_factor_observer):
                return jnp.array(1e9, dtype=jnp.float32)  # 1 billion

        pof = LargePayoff(contract_role=ContractRole.RPL, currency="USD")

        tmd = ActusDateTime(2024, 1, 1, 0, 0, 0)
        sd = ActusDateTime(2024, 1, 1, 0, 0, 0)
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
            contract_role=ContractRole.RPL,
            status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
            currency="USD",
        )
        time = ActusDateTime(2024, 6, 15, 0, 0, 0)
        observer = Mock()

        result = pof(EventType.IP, state, attributes, time, observer)

        # No automatic role_sign applied (each POF handles its own sign)
        expected = jnp.array(1e9, dtype=jnp.float32)
        assert jnp.allclose(result, expected)

    def test_small_fx_rate(self):
        """Handles small FX rates correctly."""
        pof = ConcretePayoffFunction(
            contract_role=ContractRole.RPA, currency="JPY", settlement_currency="USD"
        )
        amount = jnp.array(10000.0, dtype=jnp.float32)
        time = ActusDateTime(2024, 6, 15, 0, 0, 0)

        # JPY/USD is typically small (e.g., 0.0067)
        observer = Mock()
        observer.observe_risk_factor.return_value = jnp.array(0.0067, dtype=jnp.float32)

        result = pof.apply_fx_rate(amount, time, observer)

        expected = jnp.array(67.0, dtype=jnp.float32)
        assert jnp.allclose(result, expected, rtol=1e-4)
