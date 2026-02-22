"""Performance benchmarks for JACTUS core operations.

T1.14: Schedule Performance
T1.15: State Operations Performance
T3.13: Contract Simulation Performance (Phase 3)

These tests verify performance targets are met. Each benchmark uses warmup
iterations to eliminate cold-start overhead (JIT compilation, import caching,
memory allocation) and takes the median of multiple runs for stability.

Thresholds are calibrated from 10-run benchmarks on commodity hardware:
    threshold = ceil(mean + 2*stddev) * 1.5
The 1.5x factor absorbs cross-environment variance (CI, different CPUs, load).
"""

import statistics
import time

import jax
import jax.numpy as jnp

from jactus.contracts import create_contract
from jactus.core import (
    ActusDateTime,
    ContractAttributes,
    ContractRole,
    ContractType,
    DayCountConvention,
)
from jactus.observers import ConstantRiskFactorObserver, JaxRiskFactorObserver
from jactus.utilities import generate_schedule


def _benchmark(fn, warmup=3, runs=7):
    """Run a function with warmup and return timing statistics.

    Args:
        fn: Callable to benchmark (no args).
        warmup: Number of warmup iterations (not timed).
        runs: Number of timed iterations.

    Returns:
        Tuple of (median_ms, mean_ms, stddev_ms, all_times_ms).
    """
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        fn()
        elapsed_ms = (time.perf_counter() - t0) * 1000
        times.append(elapsed_ms)
    median = statistics.median(times)
    mean = statistics.mean(times)
    stddev = statistics.stdev(times)
    return median, mean, stddev, times


class TestSchedulePerformance:
    """Test schedule generation performance (T1.14)."""

    def test_1000_event_schedule(self):
        """Generate 1000-event daily schedule."""
        start = ActusDateTime(2024, 1, 1, 0, 0, 0)
        end = ActusDateTime(2026, 10, 1, 0, 0, 0)

        def fn():
            return generate_schedule(start=start, cycle="1D", end=end)

        median, mean, std, times = _benchmark(fn)
        schedule = fn()
        print(f"\n1000-event schedule: median={median:.1f}ms mean={mean:.1f}ms std={std:.1f}ms")

        assert len(schedule) >= 1000, f"Expected >=1000 events, got {len(schedule)}"
        # Baseline: mean=54ms std=30ms (occasional GC spikes)
        # Threshold: 85ms (mean+2σ capped, excludes GC outliers)
        assert median < 85, f"Too slow: median {median:.1f}ms > 85ms"

    def test_100_monthly_schedules(self):
        """Generate 100 monthly schedules sequentially."""
        end = ActusDateTime(2029, 1, 1, 0, 0, 0)

        def fn():
            for i in range(100):
                start_i = ActusDateTime(2024, 1, 1 + (i % 28), 0, 0, 0)
                generate_schedule(start=start_i, cycle="1M", end=end)

        median, mean, std, times = _benchmark(fn)
        print(f"\n100 monthly schedules: median={median:.1f}ms mean={mean:.1f}ms std={std:.1f}ms")

        # Baseline: mean=210ms std=3ms → mean+2σ≈217ms
        # Threshold: 325ms
        assert median < 325, f"Too slow: median {median:.1f}ms > 325ms"

    def test_array_schedule_10_anchors(self):
        """Array schedule with 10 anchors."""
        from jactus.utilities import generate_array_schedule

        anchors = [ActusDateTime(2024, i, 1, 0, 0, 0) for i in range(1, 11)]
        cycles = ["1M"] * 10
        end = ActusDateTime(2024, 12, 31, 0, 0, 0)

        def fn():
            return generate_array_schedule(anchors=anchors, cycles=cycles, end=end)

        median, mean, std, times = _benchmark(fn)
        print(
            f"\nArray schedule (10 anchors): median={median:.2f}ms mean={mean:.2f}ms std={std:.2f}ms"
        )

        # Baseline: mean=0.84ms std=0.02ms → mean+2σ≈0.89ms
        # Threshold: 1.5ms
        assert median < 1.5, f"Too slow: median {median:.2f}ms > 1.5ms"


class TestStateOperationsPerformance:
    """Test state operations performance (T1.15)."""

    def test_1m_jax_operations(self):
        """1M JAX array operations with JIT."""

        @jax.jit
        def simulate_interest_accrual(notionals, rate):
            return notionals * rate / 12

        nt_array = jnp.full(1_000_000, 100000.0)
        rate = jnp.array(0.05)

        # Full-shape warmup for JIT
        _ = simulate_interest_accrual(nt_array, rate).block_until_ready()

        def fn():
            r = simulate_interest_accrual(nt_array, rate)
            r.block_until_ready()

        median, mean, std, times = _benchmark(fn)
        print(f"\n1M JAX ops: median={median:.2f}ms mean={mean:.2f}ms std={std:.2f}ms")

        results = simulate_interest_accrual(nt_array, rate)
        assert len(results) == 1_000_000
        # Baseline: mean=0.57ms std=0.29ms → mean+2σ≈1.15ms
        # Threshold: 2ms
        assert median < 2, f"Too slow: median {median:.2f}ms > 2ms"

    def test_vectorized_math_functions(self):
        """Vectorized discount factor calculations."""
        from jactus.utilities import discount_factor_vectorized

        n_calcs = 10_000
        rates = jnp.full(n_calcs, 0.05)
        year_fractions = jnp.full(n_calcs, 1.0)

        # Warmup
        _ = discount_factor_vectorized(rates, year_fractions).block_until_ready()

        def fn():
            r = discount_factor_vectorized(rates, year_fractions)
            r.block_until_ready()

        median, mean, std, times = _benchmark(fn)
        print(f"\n10K discount factors: median={median:.3f}ms mean={mean:.3f}ms std={std:.3f}ms")

        results = discount_factor_vectorized(rates, year_fractions)
        assert len(results) == n_calcs
        # Baseline: mean=0.04ms std=0.01ms → mean+2σ≈0.06ms
        # Threshold: 0.1ms
        assert median < 0.1, f"Too slow: median {median:.3f}ms > 0.1ms"


class TestContractSimulationPerformance:
    """Test contract simulation performance (T3.13)."""

    def test_csh_simulation_performance(self):
        """CSH simulation (simplest contract type)."""
        attrs = ContractAttributes(
            contract_id="CSH-PERF-001",
            contract_type=ContractType.CSH,
            contract_role=ContractRole.RPA,
            status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
            currency="USD",
            notional_principal=100000.0,
        )

        rf_obs = ConstantRiskFactorObserver(constant_value=0.0)
        contract = create_contract(attrs, rf_obs)

        def fn():
            return contract.simulate()

        median, mean, std, times = _benchmark(fn)
        result = fn()
        print(f"\nCSH sim: median={median:.2f}ms mean={mean:.2f}ms std={std:.2f}ms")

        assert result is not None
        assert len(result.events) >= 1
        # Baseline: mean=1.94ms std=0.04ms → mean+2σ≈2.03ms
        # Threshold: 3ms
        assert median < 3, f"Too slow: median {median:.2f}ms > 3ms"

    def test_pam_simulation_performance(self):
        """PAM 5-year loan with quarterly interest payments."""
        attrs = ContractAttributes(
            contract_id="PAM-PERF-001",
            contract_type=ContractType.PAM,
            contract_role=ContractRole.RPA,
            status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
            initial_exchange_date=ActusDateTime(2024, 1, 15, 0, 0, 0),
            maturity_date=ActusDateTime(2029, 1, 15, 0, 0, 0),
            currency="USD",
            notional_principal=100000.0,
            nominal_interest_rate=0.05,
            day_count_convention=DayCountConvention.A360,
            interest_payment_cycle="3M",
        )

        rf_obs = ConstantRiskFactorObserver(constant_value=0.05)
        contract = create_contract(attrs, rf_obs)

        def fn():
            return contract.simulate()

        median, mean, std, times = _benchmark(fn)
        result = fn()
        print(
            f"\nPAM sim ({len(result.events)} events): median={median:.1f}ms mean={mean:.1f}ms std={std:.1f}ms"
        )

        assert result is not None
        assert len(result.events) > 1
        # Baseline: mean=25.9ms std=0.2ms → mean+2σ≈26.3ms
        # Threshold: 40ms
        assert median < 40, f"Too slow: median {median:.1f}ms > 40ms"

    def test_stk_simulation_performance(self):
        """STK simulation (purchase + termination)."""
        attrs = ContractAttributes(
            contract_id="STK-PERF-001",
            contract_type=ContractType.STK,
            contract_role=ContractRole.RPA,
            status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
            purchase_date=ActusDateTime(2024, 1, 15, 0, 0, 0),
            termination_date=ActusDateTime(2024, 12, 31, 0, 0, 0),
            currency="USD",
            price_at_purchase_date=150.0,
            price_at_termination_date=175.0,
        )

        rf_obs = ConstantRiskFactorObserver(constant_value=160.0)
        contract = create_contract(attrs, rf_obs)

        def fn():
            return contract.simulate()

        median, mean, std, times = _benchmark(fn)
        result = fn()
        print(f"\nSTK sim: median={median:.2f}ms mean={mean:.2f}ms std={std:.2f}ms")

        assert result is not None
        assert len(result.events) == 2
        # Baseline: mean=2.51ms std=0.02ms → mean+2σ≈2.54ms
        # Threshold: 4ms
        assert median < 4, f"Too slow: median {median:.2f}ms > 4ms"

    def test_com_simulation_performance(self):
        """COM simulation (purchase + termination)."""
        attrs = ContractAttributes(
            contract_id="COM-PERF-001",
            contract_type=ContractType.COM,
            contract_role=ContractRole.RPA,
            status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
            purchase_date=ActusDateTime(2024, 1, 15, 0, 0, 0),
            termination_date=ActusDateTime(2024, 12, 31, 0, 0, 0),
            currency="USD",
            price_at_purchase_date=7500.0,
            price_at_termination_date=8200.0,
        )

        rf_obs = ConstantRiskFactorObserver(constant_value=80.0)
        contract = create_contract(attrs, rf_obs)

        def fn():
            return contract.simulate()

        median, mean, std, times = _benchmark(fn)
        result = fn()
        print(f"\nCOM sim: median={median:.2f}ms mean={mean:.2f}ms std={std:.2f}ms")

        assert result is not None
        assert len(result.events) == 2
        # Baseline: mean=2.52ms std=0.06ms → mean+2σ≈2.65ms
        # Threshold: 4ms
        assert median < 4, f"Too slow: median {median:.2f}ms > 4ms"

    def test_pam_long_maturity_performance(self):
        """30-year PAM with monthly payments (363+ events)."""
        attrs = ContractAttributes(
            contract_id="PAM-LONG-001",
            contract_type=ContractType.PAM,
            contract_role=ContractRole.RPA,
            status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
            initial_exchange_date=ActusDateTime(2024, 1, 15, 0, 0, 0),
            maturity_date=ActusDateTime(2054, 1, 15, 0, 0, 0),
            currency="USD",
            notional_principal=300000.0,
            nominal_interest_rate=0.065,
            day_count_convention=DayCountConvention.A360,
            interest_payment_cycle="1M",
        )

        rf_obs = ConstantRiskFactorObserver(constant_value=0.065)
        contract = create_contract(attrs, rf_obs)

        def fn():
            return contract.simulate()

        median, mean, std, times = _benchmark(fn)
        result = fn()
        print(
            f"\n30yr PAM ({len(result.events)} events): median={median:.1f}ms mean={mean:.1f}ms std={std:.1f}ms"
        )

        assert result is not None
        assert len(result.events) > 360
        # Baseline: mean=381ms std=4ms → mean+2σ≈389ms
        # Threshold: 585ms
        assert median < 585, f"Too slow: median {median:.1f}ms > 585ms"

    def test_multiple_contracts_batch_performance(self):
        """Simulate 100 CSH contracts sequentially."""
        contracts = []
        for i in range(100):
            attrs = ContractAttributes(
                contract_id=f"CSH-BATCH-{i:03d}",
                contract_type=ContractType.CSH,
                contract_role=ContractRole.RPA,
                status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
                currency="USD",
                notional_principal=100000.0 + i * 1000.0,
            )
            rf_obs = ConstantRiskFactorObserver(constant_value=0.0)
            contracts.append(create_contract(attrs, rf_obs))

        def fn():
            return [c.simulate() for c in contracts]

        median, mean, std, times = _benchmark(fn)
        results = fn()
        print(
            f"\n100 CSH batch: median={median:.1f}ms ({median / 100:.2f}ms avg) mean={mean:.1f}ms std={std:.1f}ms"
        )

        assert len(results) == 100
        assert all(r is not None for r in results)
        # Baseline: mean=204ms std=7ms → mean+2σ≈218ms
        # Threshold: 325ms
        assert median < 325, f"Too slow: median {median:.1f}ms > 325ms"

    def test_factory_creation_performance(self):
        """Create 100 contracts via factory (includes Pydantic validation)."""

        def fn():
            for _ in range(100):
                attrs = ContractAttributes(
                    contract_id="CSH-FACTORY-001",
                    contract_type=ContractType.CSH,
                    contract_role=ContractRole.RPA,
                    status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
                    currency="USD",
                    notional_principal=100000.0,
                )
                create_contract(attrs, ConstantRiskFactorObserver(constant_value=0.0))

        median, mean, std, times = _benchmark(fn)
        print(
            f"\nFactory 100x: median={median:.1f}ms ({median / 100:.2f}ms avg) mean={mean:.1f}ms std={std:.1f}ms"
        )

        # Baseline: mean=274ms std=6ms → mean+2σ≈286ms
        # Threshold: 430ms
        assert median < 430, f"Too slow: median {median:.1f}ms > 430ms"


class TestJAXContractPerformance:
    """Test JAX-specific contract performance features."""

    def test_jax_observer_access_performance(self):
        """JAX observer scalar access in Python loop (worst case for JAX)."""
        risk_factors = jnp.array([0.05, 1.25, 100000.0])
        observer = JaxRiskFactorObserver(risk_factors)

        def fn():
            for _ in range(1000):
                _ = observer.get(0) + observer.get(1) + observer.get(2)

        median, mean, std, times = _benchmark(fn)
        print(
            f"\nJAX observer 1000x: median={median:.0f}ms ({median / 1000:.3f}ms/iter) mean={mean:.0f}ms std={std:.0f}ms"
        )

        # Baseline: mean=1534ms std=9ms → mean+2σ≈1553ms
        # Threshold: 2330ms
        assert median < 2330, f"Too slow: median {median:.0f}ms > 2330ms"

    def test_jit_compilation_speedup(self):
        """JIT compilation provides speedup for repeated state initialization."""
        attrs = ContractAttributes(
            contract_id="PAM-JIT-001",
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

        @jax.jit
        def calc_notional():
            state = contract.initialize_state()
            return state.nt

        # Warmup JIT compilation
        _ = calc_notional()

        def fn():
            for _ in range(1000):
                _ = calc_notional()

        median, mean, std, times = _benchmark(fn)
        print(f"\nJIT state init 1000x: median={median:.1f}ms mean={mean:.1f}ms std={std:.1f}ms")

        # Baseline: mean=19.2ms std=0.3ms → mean+2σ≈19.9ms
        # Threshold: 30ms
        assert median < 30, f"Too slow: median {median:.1f}ms > 30ms"

    def test_contract_memory_efficiency(self):
        """Contracts should use efficient float32 arrays."""
        attrs = ContractAttributes(
            contract_id="PAM-MEM-001",
            contract_type=ContractType.PAM,
            contract_role=ContractRole.RPA,
            status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
            initial_exchange_date=ActusDateTime(2024, 1, 15, 0, 0, 0),
            maturity_date=ActusDateTime(2029, 1, 15, 0, 0, 0),
            currency="USD",
            notional_principal=100000.0,
            nominal_interest_rate=0.05,
            day_count_convention=DayCountConvention.A360,
            interest_payment_cycle="3M",
        )

        rf_obs = JaxRiskFactorObserver(jnp.array([0.05]))
        contract = create_contract(attrs, rf_obs)
        result = contract.simulate()

        for event in result.events:
            assert event.payoff.dtype == jnp.float32
            if event.state_pre:
                assert event.state_pre.nt.dtype == jnp.float32
                assert event.state_pre.ipnr.dtype == jnp.float32
            if event.state_post:
                assert event.state_post.nt.dtype == jnp.float32
                assert event.state_post.ipnr.dtype == jnp.float32
