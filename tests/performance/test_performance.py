"""Performance benchmarks for JACTUS core operations.

T1.14: Schedule Performance
T1.15: State Operations Performance
T3.13: Contract Simulation Performance (Phase 3)

These tests verify performance targets are met.
"""

import time

import jax
import jax.numpy as jnp

from jactus.contracts import create_contract
from jactus.core import (
    ActusDateTime,
    ContractAttributes,
    ContractRole,
    ContractState,
    ContractType,
    DayCountConvention,
)
from jactus.observers import ConstantRiskFactorObserver, JaxRiskFactorObserver
from jactus.utilities import generate_schedule


class TestSchedulePerformance:
    """Test schedule generation performance (T1.14)."""

    def test_1000_event_schedule_under_50ms(self):
        """Generate 1000-event schedule in under 50ms."""
        start = ActusDateTime(2024, 1, 1, 0, 0, 0)
        end = ActusDateTime(2026, 10, 1, 0, 0, 0)  # ~3 years daily

        start_time = time.perf_counter()
        schedule = generate_schedule(start=start, cycle="1D", end=end)
        elapsed = (time.perf_counter() - start_time) * 1000  # Convert to ms

        print(f"\nGenerated {len(schedule)} dates in {elapsed:.2f}ms")
        assert len(schedule) >= 1000, f"Expected >=1000 events, got {len(schedule)}"
        assert elapsed < 50, f"Too slow: {elapsed:.2f}ms > 50ms"

    def test_100_monthly_schedules_parallel(self):
        """Generate 100 schedules in under 150ms total."""
        start = ActusDateTime(2024, 1, 1, 0, 0, 0)
        end = ActusDateTime(2029, 1, 1, 0, 0, 0)

        start_time = time.perf_counter()
        schedules = []
        for i in range(100):
            # Slightly offset starts to avoid caching
            start_i = ActusDateTime(2024, 1, 1 + (i % 28), 0, 0, 0)
            schedule = generate_schedule(start=start_i, cycle="1M", end=end)
            schedules.append(schedule)
        elapsed = (time.perf_counter() - start_time) * 1000

        print(f"\nGenerated {len(schedules)} schedules in {elapsed:.2f}ms")
        # Allow 150ms to account for coverage overhead
        assert elapsed < 150, f"Too slow: {elapsed:.2f}ms > 150ms"

    def test_array_schedule_10_anchors(self):
        """Array schedule with 10 anchors in under 30ms."""
        # 10 anchors, each 1 month apart
        anchors = [ActusDateTime(2024, i, 1, 0, 0, 0) for i in range(1, 11)]
        cycles = ["1M"] * 10
        end = ActusDateTime(2024, 12, 31, 0, 0, 0)

        from jactus.utilities import generate_array_schedule

        start_time = time.perf_counter()
        schedule = generate_array_schedule(anchors=anchors, cycles=cycles, end=end)
        elapsed = (time.perf_counter() - start_time) * 1000

        print(f"\nGenerated array schedule with {len(schedule)} dates in {elapsed:.2f}ms")
        assert elapsed < 30, f"Too slow: {elapsed:.2f}ms > 30ms"


class TestStateOperationsPerformance:
    """Test state operations performance (T1.15)."""

    def test_1m_jax_operations_under_20ms(self):
        """1M JAX array operations should complete in under 20ms with JIT."""
        import jax

        # Define a simple JIT-compiled function
        @jax.jit
        def simulate_interest_accrual(notionals, rate):
            """Vectorized interest calculation."""
            return notionals * rate / 12  # Monthly interest

        # Create array of 1M notionals
        nt_array = jnp.full(1_000_000, 100000.0)
        rate = jnp.array(0.05)

        # Warm up JIT compilation
        _ = simulate_interest_accrual(nt_array[:1000], rate)

        # Time the operation
        start_time = time.perf_counter()
        results = simulate_interest_accrual(nt_array, rate)
        # Block until computation completes
        results.block_until_ready()
        elapsed = (time.perf_counter() - start_time) * 1000

        print(f"\n1M array operations completed in {elapsed:.2f}ms")
        assert elapsed < 20, f"Too slow: {elapsed:.2f}ms > 20ms"
        assert len(results) == 1_000_000

    def test_vectorized_math_functions(self):
        """Vectorized math functions should handle large arrays efficiently."""
        from jactus.utilities import discount_factor_vectorized

        # Create arrays for 10K calculations
        n_calcs = 10_000
        rates = jnp.full(n_calcs, 0.05)
        year_fractions = jnp.full(n_calcs, 1.0)

        # Time the vectorized operation
        start_time = time.perf_counter()
        results = discount_factor_vectorized(rates, year_fractions)
        results.block_until_ready()
        elapsed = (time.perf_counter() - start_time) * 1000

        print(f"\n10K discount factor calculations in {elapsed:.2f}ms")
        assert len(results) == n_calcs
        assert elapsed < 20, f"Too slow: {elapsed:.2f}ms > 20ms"


class TestContractSimulationPerformance:
    """Test contract simulation performance (T3.13)."""

    def test_csh_simulation_performance(self):
        """CSH simulation should be very fast (< 10ms)."""
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

        start_time = time.perf_counter()
        result = contract.simulate()
        elapsed = (time.perf_counter() - start_time) * 1000

        print(f"\nCSH simulation completed in {elapsed:.2f}ms")
        assert result is not None
        assert len(result.events) >= 1
        assert elapsed < 10, f"Too slow: {elapsed:.2f}ms > 10ms"

    def test_pam_simulation_performance(self):
        """PAM 5-year loan with quarterly payments should be fast (< 50ms)."""
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

        start_time = time.perf_counter()
        result = contract.simulate()
        elapsed = (time.perf_counter() - start_time) * 1000

        print(f"\nPAM simulation ({len(result.events)} events) in {elapsed:.2f}ms")
        assert result is not None
        assert len(result.events) > 1
        assert elapsed < 50, f"Too slow: {elapsed:.2f}ms > 50ms"

    def test_stk_simulation_performance(self):
        """STK simulation should be very fast (< 10ms)."""
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

        start_time = time.perf_counter()
        result = contract.simulate()
        elapsed = (time.perf_counter() - start_time) * 1000

        print(f"\nSTK simulation completed in {elapsed:.2f}ms")
        assert result is not None
        assert len(result.events) == 2
        assert elapsed < 10, f"Too slow: {elapsed:.2f}ms > 10ms"

    def test_com_simulation_performance(self):
        """COM simulation should be very fast (< 10ms)."""
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

        start_time = time.perf_counter()
        result = contract.simulate()
        elapsed = (time.perf_counter() - start_time) * 1000

        print(f"\nCOM simulation completed in {elapsed:.2f}ms")
        assert result is not None
        assert len(result.events) == 2
        assert elapsed < 10, f"Too slow: {elapsed:.2f}ms > 10ms"

    def test_pam_long_maturity_performance(self):
        """30-year PAM with monthly payments should complete in < 500ms."""
        attrs = ContractAttributes(
            contract_id="PAM-LONG-001",
            contract_type=ContractType.PAM,
            contract_role=ContractRole.RPA,
            status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
            initial_exchange_date=ActusDateTime(2024, 1, 15, 0, 0, 0),
            maturity_date=ActusDateTime(2054, 1, 15, 0, 0, 0),  # 30 years
            currency="USD",
            notional_principal=300000.0,
            nominal_interest_rate=0.065,
            day_count_convention=DayCountConvention.A360,
            interest_payment_cycle="1M",  # Monthly payments
        )

        rf_obs = ConstantRiskFactorObserver(constant_value=0.065)
        contract = create_contract(attrs, rf_obs)

        start_time = time.perf_counter()
        result = contract.simulate()
        elapsed = (time.perf_counter() - start_time) * 1000

        print(f"\n30-year PAM ({len(result.events)} events) in {elapsed:.2f}ms")
        assert result is not None
        assert len(result.events) > 360  # 30 years * 12 months
        assert elapsed < 500, f"Too slow: {elapsed:.2f}ms > 500ms"

    def test_multiple_contracts_batch_performance(self):
        """100 contracts should simulate in < 500ms."""
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
            contract = create_contract(attrs, rf_obs)
            contracts.append(contract)

        start_time = time.perf_counter()
        results = [c.simulate() for c in contracts]
        elapsed = (time.perf_counter() - start_time) * 1000

        print(f"\n100 CSH contracts simulated in {elapsed:.2f}ms ({elapsed/100:.2f}ms avg)")
        assert len(results) == 100
        assert all(r is not None for r in results)
        assert elapsed < 500, f"Too slow: {elapsed:.2f}ms > 500ms"

    def test_factory_creation_performance(self):
        """Contract factory should create contracts efficiently (< 1ms each)."""
        start_time = time.perf_counter()

        for _ in range(100):
            attrs = ContractAttributes(
                contract_id="CSH-FACTORY-001",
                contract_type=ContractType.CSH,
                contract_role=ContractRole.RPA,
                status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
                currency="USD",
                notional_principal=100000.0,
            )

            rf_obs = ConstantRiskFactorObserver(constant_value=0.0)
            contract = create_contract(attrs, rf_obs)

        elapsed = (time.perf_counter() - start_time) * 1000

        print(f"\n100 contract creations in {elapsed:.2f}ms ({elapsed/100:.2f}ms avg)")
        assert elapsed < 100, f"Too slow: {elapsed:.2f}ms > 100ms"


class TestJAXContractPerformance:
    """Test JAX-specific contract performance features."""

    def test_jax_observer_access_performance(self):
        """JAX observer should provide reasonable array access performance."""
        risk_factors = jnp.array([0.05, 1.25, 100000.0])
        observer = JaxRiskFactorObserver(risk_factors)

        start_time = time.perf_counter()
        for _ in range(1000):
            _ = observer.get(0) + observer.get(1) + observer.get(2)
        elapsed = (time.perf_counter() - start_time) * 1000

        print(f"\n1000 JAX observer accesses in {elapsed:.2f}ms ({elapsed/1000:.3f}ms avg)")
        # Allow 500ms for 1000 accesses (0.5ms per access)
        assert elapsed < 500, f"Too slow: {elapsed:.2f}ms > 500ms"

    def test_jit_compilation_speedup(self):
        """JIT compilation should provide speedup for repeated operations."""
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

        # Warmup JIT
        _ = calc_notional()

        # Time JIT version
        start_time = time.perf_counter()
        for _ in range(1000):
            _ = calc_notional()
        elapsed = (time.perf_counter() - start_time) * 1000

        print(f"\n1000 JIT'd state initializations in {elapsed:.2f}ms")
        assert elapsed < 100, f"Too slow: {elapsed:.2f}ms > 100ms"

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

        # Verify all arrays are float32 (memory efficient)
        for event in result.events:
            assert event.payoff.dtype == jnp.float32
            if event.state_pre:
                assert event.state_pre.nt.dtype == jnp.float32
                assert event.state_pre.ipnr.dtype == jnp.float32
            if event.state_post:
                assert event.state_post.nt.dtype == jnp.float32
                assert event.state_post.ipnr.dtype == jnp.float32
