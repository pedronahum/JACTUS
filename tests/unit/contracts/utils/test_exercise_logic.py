"""Unit tests for exercise logic utilities."""

import jax
import jax.numpy as jnp
import pytest

from jactus.contracts.utils.exercise_logic import (
    ExerciseDecision,
    calculate_intrinsic_value,
    calculate_option_greeks,
    should_exercise,
)


class TestCalculateIntrinsicValue:
    """Test intrinsic value calculations for different option types."""

    def test_call_option_in_the_money(self):
        """Test call option intrinsic value when in-the-money."""
        # Spot = 105, Strike = 100, ITM by 5
        iv = calculate_intrinsic_value("C", 105.0, 100.0)
        assert float(iv) == pytest.approx(5.0, abs=0.01)

    def test_call_option_at_the_money(self):
        """Test call option intrinsic value when at-the-money."""
        # Spot = Strike = 100
        iv = calculate_intrinsic_value("C", 100.0, 100.0)
        assert float(iv) == pytest.approx(0.0, abs=0.01)

    def test_call_option_out_of_the_money(self):
        """Test call option intrinsic value when out-of-the-money."""
        # Spot = 95, Strike = 100, OTM
        iv = calculate_intrinsic_value("C", 95.0, 100.0)
        assert float(iv) == pytest.approx(0.0, abs=0.01)

    def test_put_option_in_the_money(self):
        """Test put option intrinsic value when in-the-money."""
        # Spot = 95, Strike = 100, ITM by 5
        iv = calculate_intrinsic_value("P", 95.0, 100.0)
        assert float(iv) == pytest.approx(5.0, abs=0.01)

    def test_put_option_at_the_money(self):
        """Test put option intrinsic value when at-the-money."""
        # Spot = Strike = 100
        iv = calculate_intrinsic_value("P", 100.0, 100.0)
        assert float(iv) == pytest.approx(0.0, abs=0.01)

    def test_put_option_out_of_the_money(self):
        """Test put option intrinsic value when out-of-the-money."""
        # Spot = 105, Strike = 100, OTM
        iv = calculate_intrinsic_value("P", 105.0, 100.0)
        assert float(iv) == pytest.approx(0.0, abs=0.01)

    def test_collar_option_both_in_the_money(self):
        """Test collar option when both strikes are in-the-money."""
        # Spot = 105, K1 = 100, K2 = 110
        # Call at 100: max(105-100, 0) = 5
        # Put at 110: max(110-105, 0) = 5
        # Total = 10
        iv = calculate_intrinsic_value("CP", 105.0, 100.0, 110.0)
        assert float(iv) == pytest.approx(10.0, abs=0.01)

    def test_collar_option_call_only(self):
        """Test collar option when only call is in-the-money."""
        # Spot = 105, K1 = 100, K2 = 95
        # Call at 100: max(105-100, 0) = 5
        # Put at 95: max(95-105, 0) = 0
        # Total = 5
        iv = calculate_intrinsic_value("CP", 105.0, 100.0, 95.0)
        assert float(iv) == pytest.approx(5.0, abs=0.01)

    def test_collar_option_put_only(self):
        """Test collar option when only put is in-the-money."""
        # Spot = 95, K1 = 100, K2 = 110
        # Call at 100: max(95-100, 0) = 0
        # Put at 110: max(110-95, 0) = 15
        # Total = 15
        iv = calculate_intrinsic_value("CP", 95.0, 100.0, 110.0)
        assert float(iv) == pytest.approx(15.0, abs=0.01)

    def test_collar_requires_strike_2(self):
        """Test that collar option requires strike_2 parameter."""
        with pytest.raises(ValueError, match="strike_2 required"):
            calculate_intrinsic_value("CP", 105.0, 100.0)

    def test_call_alias_uppercase(self):
        """Test that 'CALL' alias works."""
        iv = calculate_intrinsic_value("CALL", 105.0, 100.0)
        assert float(iv) == pytest.approx(5.0, abs=0.01)

    def test_put_alias_uppercase(self):
        """Test that 'PUT' alias works."""
        iv = calculate_intrinsic_value("PUT", 95.0, 100.0)
        assert float(iv) == pytest.approx(5.0, abs=0.01)

    def test_collar_alias_uppercase(self):
        """Test that 'COLLAR' alias works."""
        iv = calculate_intrinsic_value("COLLAR", 105.0, 100.0, 110.0)
        assert float(iv) == pytest.approx(10.0, abs=0.01)

    def test_invalid_option_type(self):
        """Test that invalid option type raises ValueError."""
        with pytest.raises(ValueError, match="Unknown option type"):
            calculate_intrinsic_value("X", 105.0, 100.0)

    def test_jax_array_inputs(self):
        """Test that JAX array inputs work correctly."""
        spot = jnp.array(105.0)
        strike = jnp.array(100.0)
        iv = calculate_intrinsic_value("C", spot, strike)
        assert isinstance(iv, jnp.ndarray)
        assert float(iv) == pytest.approx(5.0, abs=0.01)

    def test_vectorized_calculation(self):
        """Test that intrinsic value can be vectorized."""
        spots = jnp.array([95.0, 100.0, 105.0, 110.0])
        strike = 100.0

        # Vectorize over spot prices
        iv_vect = jax.vmap(lambda s: calculate_intrinsic_value("C", s, strike))(spots)

        expected = jnp.array([0.0, 0.0, 5.0, 10.0])
        assert jnp.allclose(iv_vect, expected, atol=0.01)


class TestShouldExercise:
    """Test exercise decision logic."""

    def test_should_exercise_above_threshold(self):
        """Test exercise decision when intrinsic value > threshold."""
        should_ex = should_exercise(5.0, threshold=0.0)
        assert bool(should_ex) is True

    def test_should_not_exercise_at_threshold(self):
        """Test no exercise when intrinsic value = threshold."""
        should_ex = should_exercise(0.0, threshold=0.0)
        assert bool(should_ex) is False

    def test_should_not_exercise_below_threshold(self):
        """Test no exercise when intrinsic value < threshold."""
        should_ex = should_exercise(-1.0, threshold=0.0)
        assert bool(should_ex) is False

    def test_custom_threshold(self):
        """Test exercise with custom threshold."""
        # Only exercise if intrinsic > 2.0
        should_ex_1 = should_exercise(1.5, threshold=2.0)
        should_ex_2 = should_exercise(2.5, threshold=2.0)

        assert bool(should_ex_1) is False
        assert bool(should_ex_2) is True

    def test_jax_array_input(self):
        """Test that JAX array input works."""
        iv = jnp.array(5.0)
        should_ex = should_exercise(iv)
        assert isinstance(should_ex, jnp.ndarray)
        assert bool(should_ex) is True


class TestExerciseDecision:
    """Test ExerciseDecision enum."""

    def test_exercise_decision_values(self):
        """Test that ExerciseDecision enum has correct values."""
        assert ExerciseDecision.EXERCISE.value == "EXERCISE"
        assert ExerciseDecision.HOLD.value == "HOLD"
        assert ExerciseDecision.EXPIRED.value == "EXPIRED"

    def test_exercise_decision_membership(self):
        """Test ExerciseDecision enum membership."""
        assert ExerciseDecision.EXERCISE in ExerciseDecision
        assert ExerciseDecision.HOLD in ExerciseDecision
        assert ExerciseDecision.EXPIRED in ExerciseDecision


class TestCalculateOptionGreeks:
    """Test option Greeks calculation using JAX autodiff."""

    def test_greeks_call_option(self):
        """Test Greeks calculation for call option."""
        greeks = calculate_option_greeks("C", 105.0, 100.0, 0.2, 1.0, 0.05)

        # Should have delta and intrinsic
        assert "delta" in greeks
        assert "intrinsic" in greeks

        # Intrinsic value should be 5.0
        assert greeks["intrinsic"] == pytest.approx(5.0, abs=0.01)

    def test_greeks_put_option(self):
        """Test Greeks calculation for put option."""
        greeks = calculate_option_greeks("P", 95.0, 100.0, 0.2, 1.0, 0.05)

        # Should have delta and intrinsic
        assert "delta" in greeks
        assert "intrinsic" in greeks

        # Intrinsic value should be 5.0
        assert greeks["intrinsic"] == pytest.approx(5.0, abs=0.01)

    def test_greeks_at_the_money(self):
        """Test Greeks when option is at-the-money."""
        greeks = calculate_option_greeks("C", 100.0, 100.0, 0.2, 1.0, 0.05)

        # Intrinsic value should be 0.0
        assert greeks["intrinsic"] == pytest.approx(0.0, abs=0.01)


class TestJAXCompatibility:
    """Test JAX compatibility (jit, grad, vmap)."""

    def test_jit_compilation(self):
        """Test that intrinsic value can be JIT compiled."""

        @jax.jit
        def jitted_iv(spot, strike):
            return calculate_intrinsic_value("C", spot, strike)

        iv = jitted_iv(jnp.array(105.0), jnp.array(100.0))
        assert float(iv) == pytest.approx(5.0, abs=0.01)

    def test_gradient_calculation(self):
        """Test that gradient (delta) can be calculated."""

        def payoff(spot):
            return calculate_intrinsic_value("C", spot, 100.0)

        # Calculate delta at spot = 105
        delta_fn = jax.grad(payoff)
        delta = delta_fn(105.0)

        # Delta should be 0 for intrinsic value (discontinuous at strike)
        # or close to 0 away from strike
        assert isinstance(delta, jnp.ndarray)

    def test_vmap_vectorization(self):
        """Test that vmap can vectorize over inputs."""
        spots = jnp.array([95.0, 100.0, 105.0])

        # Vectorize calculate_intrinsic_value over spots
        iv_vec = jax.vmap(lambda s: calculate_intrinsic_value("C", s, 100.0))(spots)

        expected = jnp.array([0.0, 0.0, 5.0])
        assert jnp.allclose(iv_vec, expected, atol=0.01)
