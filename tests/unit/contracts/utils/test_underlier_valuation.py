"""Unit tests for underlier valuation utilities."""

import jax.numpy as jnp
import pytest

from jactus.contracts.utils.underlier_valuation import (
    calculate_forward_price,
    calculate_present_value,
    get_underlier_market_value,
    get_underlier_price_path,
)
from jactus.core import ActusDateTime
from jactus.observers.risk_factor import ConstantRiskFactorObserver


class TestGetUnderlierMarketValue:
    """Test underlier market value observation."""

    def test_get_stock_price(self):
        """Test getting stock price from market."""
        rf_obs = ConstantRiskFactorObserver(constant_value=150.0)
        time = ActusDateTime(2024, 1, 15, 0, 0, 0)

        value = get_underlier_market_value("AAPL", time, rf_obs)

        assert isinstance(value, jnp.ndarray)
        assert float(value) == pytest.approx(150.0, abs=0.01)

    def test_get_commodity_price(self):
        """Test getting commodity price from market."""
        rf_obs = ConstantRiskFactorObserver(constant_value=1800.0)
        time = ActusDateTime(2024, 1, 15, 0, 0, 0)

        value = get_underlier_market_value("GC", time, rf_obs)

        assert isinstance(value, jnp.ndarray)
        assert float(value) == pytest.approx(1800.0, abs=0.01)

    def test_get_index_price(self):
        """Test getting index price from market."""
        rf_obs = ConstantRiskFactorObserver(constant_value=4500.0)
        time = ActusDateTime(2024, 1, 15, 0, 0, 0)

        value = get_underlier_market_value("SPX", time, rf_obs)

        assert isinstance(value, jnp.ndarray)
        assert float(value) == pytest.approx(4500.0, abs=0.01)

    def test_different_market_codes(self):
        """Test that different market codes can be observed."""
        rf_obs = ConstantRiskFactorObserver(constant_value=100.0)
        time = ActusDateTime(2024, 1, 15, 0, 0, 0)

        codes = ["AAPL", "GLD", "SPX", "EURUSD"]
        for code in codes:
            value = get_underlier_market_value(code, time, rf_obs)
            assert isinstance(value, jnp.ndarray)

    def test_non_string_reference_raises_error(self):
        """Test that non-string reference raises NotImplementedError."""
        rf_obs = ConstantRiskFactorObserver(constant_value=100.0)
        time = ActusDateTime(2024, 1, 15, 0, 0, 0)

        # Contract references not yet implemented
        with pytest.raises(NotImplementedError, match="not yet implemented"):
            get_underlier_market_value(12345, time, rf_obs)

    def test_returns_jax_array(self):
        """Test that return value is JAX array."""
        rf_obs = ConstantRiskFactorObserver(constant_value=150.0)
        time = ActusDateTime(2024, 1, 15, 0, 0, 0)

        value = get_underlier_market_value("AAPL", time, rf_obs)

        assert isinstance(value, jnp.ndarray)
        assert value.dtype == jnp.float32


class TestGetUnderlierPricePath:
    """Test underlier price path observation."""

    def test_price_path_single_time(self):
        """Test price path with single time point."""
        rf_obs = ConstantRiskFactorObserver(constant_value=150.0)
        times = [ActusDateTime(2024, 1, 15, 0, 0, 0)]

        prices = get_underlier_price_path("AAPL", times, rf_obs)

        assert isinstance(prices, jnp.ndarray)
        assert len(prices) == 1
        assert float(prices[0]) == pytest.approx(150.0, abs=0.01)

    def test_price_path_multiple_times(self):
        """Test price path with multiple time points."""
        rf_obs = ConstantRiskFactorObserver(constant_value=150.0)
        times = [
            ActusDateTime(2024, 1, 1, 0, 0, 0),
            ActusDateTime(2024, 2, 1, 0, 0, 0),
            ActusDateTime(2024, 3, 1, 0, 0, 0),
        ]

        prices = get_underlier_price_path("AAPL", times, rf_obs)

        assert isinstance(prices, jnp.ndarray)
        assert len(prices) == 3
        # All prices should be 150 (constant observer)
        assert jnp.allclose(prices, 150.0, atol=0.01)

    def test_price_path_empty_times(self):
        """Test price path with empty time list."""
        rf_obs = ConstantRiskFactorObserver(constant_value=150.0)
        times = []

        prices = get_underlier_price_path("AAPL", times, rf_obs)

        assert isinstance(prices, jnp.ndarray)
        assert len(prices) == 0

    def test_price_path_returns_array(self):
        """Test that price path returns JAX array."""
        rf_obs = ConstantRiskFactorObserver(constant_value=150.0)
        times = [
            ActusDateTime(2024, 1, 1, 0, 0, 0),
            ActusDateTime(2024, 2, 1, 0, 0, 0),
        ]

        prices = get_underlier_price_path("AAPL", times, rf_obs)

        assert isinstance(prices, jnp.ndarray)
        assert prices.dtype == jnp.float32


class TestCalculateForwardPrice:
    """Test forward price calculation."""

    def test_forward_price_zero_time(self):
        """Test forward price with zero time to maturity."""
        spot = 100.0
        rate = 0.05
        time = 0.0

        fwd = calculate_forward_price(spot, rate, time)

        # F = S when T = 0
        assert float(fwd) == pytest.approx(100.0, abs=0.01)

    def test_forward_price_one_year(self):
        """Test forward price for one year."""
        spot = 100.0
        rate = 0.05  # 5%
        time = 1.0

        fwd = calculate_forward_price(spot, rate, time)

        # F = 100 * exp(0.05 * 1) ≈ 105.13
        expected = spot * jnp.exp(rate * time)
        assert float(fwd) == pytest.approx(float(expected), abs=0.01)

    def test_forward_price_with_dividend(self):
        """Test forward price with dividend yield."""
        spot = 100.0
        rate = 0.05  # 5%
        time = 1.0
        div_yield = 0.02  # 2% dividend yield

        fwd = calculate_forward_price(spot, rate, time, div_yield)

        # F = 100 * exp((0.05 - 0.02) * 1) ≈ 103.05
        expected = spot * jnp.exp((rate - div_yield) * time)
        assert float(fwd) == pytest.approx(float(expected), abs=0.01)

    def test_forward_price_six_months(self):
        """Test forward price for six months."""
        spot = 100.0
        rate = 0.05
        time = 0.5  # 6 months

        fwd = calculate_forward_price(spot, rate, time)

        # F = 100 * exp(0.05 * 0.5) ≈ 102.53
        expected = spot * jnp.exp(rate * time)
        assert float(fwd) == pytest.approx(float(expected), abs=0.01)

    def test_forward_price_high_dividend(self):
        """Test forward price with high dividend yield."""
        spot = 100.0
        rate = 0.05
        time = 1.0
        div_yield = 0.08  # 8% dividend (higher than risk-free)

        fwd = calculate_forward_price(spot, rate, time, div_yield)

        # F = 100 * exp((0.05 - 0.08) * 1) ≈ 97.04
        # Forward price < spot when dividend > risk-free
        assert float(fwd) < spot

    def test_forward_price_jax_arrays(self):
        """Test forward price with JAX array inputs."""
        spot = jnp.array(100.0)
        rate = jnp.array(0.05)
        time = jnp.array(1.0)

        fwd = calculate_forward_price(spot, rate, time)

        assert isinstance(fwd, jnp.ndarray)
        assert float(fwd) > 100.0  # Should be higher than spot

    def test_forward_price_vectorized(self):
        """Test that forward price can be vectorized."""
        spots = jnp.array([95.0, 100.0, 105.0, 110.0])
        rate = 0.05
        time = 1.0

        # Vectorize over spot prices
        import jax

        fwd_vec = jax.vmap(lambda s: calculate_forward_price(s, rate, time))(spots)

        assert len(fwd_vec) == 4
        # All forward prices should be higher than spots
        assert jnp.all(fwd_vec > spots)


class TestCalculatePresentValue:
    """Test present value calculation."""

    def test_pv_zero_time(self):
        """Test present value with zero time."""
        cf = 100.0
        rate = 0.05
        time = 0.0

        pv = calculate_present_value(cf, rate, time)

        # PV = CF when T = 0
        assert float(pv) == pytest.approx(100.0, abs=0.01)

    def test_pv_one_year(self):
        """Test present value for one year."""
        cf = 100.0
        rate = 0.05
        time = 1.0

        pv = calculate_present_value(cf, rate, time)

        # PV = 100 * exp(-0.05 * 1) ≈ 95.12
        expected = cf * jnp.exp(-rate * time)
        assert float(pv) == pytest.approx(float(expected), abs=0.01)

    def test_pv_six_months(self):
        """Test present value for six months."""
        cf = 100.0
        rate = 0.05
        time = 0.5

        pv = calculate_present_value(cf, rate, time)

        # PV = 100 * exp(-0.05 * 0.5) ≈ 97.53
        expected = cf * jnp.exp(-rate * time)
        assert float(pv) == pytest.approx(float(expected), abs=0.01)

    def test_pv_high_rate(self):
        """Test present value with high discount rate."""
        cf = 100.0
        rate = 0.15  # 15%
        time = 1.0

        pv = calculate_present_value(cf, rate, time)

        # PV = 100 * exp(-0.15 * 1) ≈ 86.07
        # Higher rate means lower PV
        assert float(pv) < 90.0

    def test_pv_long_maturity(self):
        """Test present value with long maturity."""
        cf = 100.0
        rate = 0.05
        time = 10.0  # 10 years

        pv = calculate_present_value(cf, rate, time)

        # PV = 100 * exp(-0.05 * 10) ≈ 60.65
        expected = cf * jnp.exp(-rate * time)
        assert float(pv) == pytest.approx(float(expected), abs=0.01)

    def test_pv_jax_arrays(self):
        """Test present value with JAX array inputs."""
        cf = jnp.array(100.0)
        rate = jnp.array(0.05)
        time = jnp.array(1.0)

        pv = calculate_present_value(cf, rate, time)

        assert isinstance(pv, jnp.ndarray)
        assert float(pv) < 100.0  # PV should be less than future CF

    def test_pv_vectorized(self):
        """Test that present value can be vectorized."""
        cfs = jnp.array([90.0, 100.0, 110.0, 120.0])
        rate = 0.05
        time = 1.0

        # Vectorize over cashflows
        import jax

        pv_vec = jax.vmap(lambda c: calculate_present_value(c, rate, time))(cfs)

        assert len(pv_vec) == 4
        # All PVs should be less than CFs
        assert jnp.all(pv_vec < cfs)


class TestJAXCompatibility:
    """Test JAX compatibility (jit, grad, vmap)."""

    def test_forward_price_jit(self):
        """Test that forward price can be JIT compiled."""
        import jax

        @jax.jit
        def jitted_fwd(spot, rate, time):
            return calculate_forward_price(spot, rate, time)

        fwd = jitted_fwd(jnp.array(100.0), jnp.array(0.05), jnp.array(1.0))
        assert float(fwd) > 100.0

    def test_pv_jit(self):
        """Test that present value can be JIT compiled."""
        import jax

        @jax.jit
        def jitted_pv(cf, rate, time):
            return calculate_present_value(cf, rate, time)

        pv = jitted_pv(jnp.array(100.0), jnp.array(0.05), jnp.array(1.0))
        assert float(pv) < 100.0

    def test_forward_price_grad(self):
        """Test that gradient of forward price can be calculated."""
        import jax

        def fwd_spot(spot):
            return calculate_forward_price(spot, 0.05, 1.0)

        # Calculate derivative w.r.t. spot
        grad_fn = jax.grad(fwd_spot)
        derivative = grad_fn(100.0)

        # dF/dS = exp((r - q) * T) ≈ 1.0513 for r=0.05, T=1
        expected = jnp.exp(0.05 * 1.0)
        assert float(derivative) == pytest.approx(float(expected), abs=0.01)

    def test_pv_grad(self):
        """Test that gradient of PV can be calculated."""
        import jax

        def pv_rate(rate):
            return calculate_present_value(100.0, rate, 1.0)

        # Calculate derivative w.r.t. rate
        grad_fn = jax.grad(pv_rate)
        derivative = grad_fn(0.05)

        # dPV/dr = -T * CF * exp(-r * T) = -1 * 100 * exp(-0.05) ≈ -95.12
        assert float(derivative) < 0  # PV decreases with rate
