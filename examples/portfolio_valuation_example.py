#!/usr/bin/env python3
"""
Portfolio Valuation Example — Present Value of Millions of PAM Loans
====================================================================

This example demonstrates how to efficiently value a large portfolio of
PAM (Principal at Maturity) loans using JACTUS, with a focus on performance
optimizations for production-scale portfolios.

Key Techniques:
--------------
1. **status_date as valuation date** — Setting status_date to the valuation
   date causes JACTUS to skip all historical events. Only future cash flows
   are generated, dramatically reducing work per contract.

2. **JAX vectorized PV** — After extracting cash flows, we use
   `present_value_vectorized` (JIT-compiled) to discount them in a single
   batched operation per contract rather than a Python loop.

3. **Parallel simulation** — Python's `concurrent.futures` parallelizes the
   per-contract simulation loop across CPU cores, since the ACTUS event
   engine is Python-level (not yet JIT-compiled).

4. **block_until_ready()** — Ensures JAX async dispatch doesn't hide latency
   when benchmarking, and forces results to be materialized.

Performance Analysis:
--------------------
The simulation loop (event schedule generation, POF/STF application) is
Python-level code, so `jax.jit` and `jax.vmap` cannot accelerate it today.
The main levers are:

  - Fewer events per contract (status_date filtering) — O(n) improvement
  - CPU parallelism for simulation (ProcessPoolExecutor) — ~linear in cores
  - JAX vectorized discounting for PV — significant for large cashflow arrays
  - JAX `block_until_ready()` — accurate benchmarking

Future JAX optimizations (not yet implemented in JACTUS) would require
making the entire simulation loop a pure function operating on JAX arrays,
enabling `jax.vmap` over contract parameter vectors.

Example: 1,000 PAM loans originated 2020–2024, valued as of 2025-06-01
"""

import multiprocessing
import random
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import jax.numpy as jnp

from jactus.contracts import create_contract
from jactus.core import (
    ActusDateTime,
    ContractAttributes,
    ContractRole,
    ContractType,
    DayCountConvention,
)
from jactus.observers import ConstantRiskFactorObserver
from jactus.utilities import present_value_vectorized, year_fraction


# ---------------------------------------------------------------------------
# 1. Portfolio generation
# ---------------------------------------------------------------------------


def generate_random_portfolio(
    n_loans: int,
    valuation_date: ActusDateTime,
    seed: int = 42,
) -> list[dict]:
    """Generate a synthetic portfolio of PAM loans.

    Creates loans with random origination dates (2020–2024), maturities
    (3–10 years from origination), rates (3–8%), and notionals ($50k–$500k).

    Args:
        n_loans: Number of loans to generate.
        valuation_date: The valuation / status date.
        seed: Random seed for reproducibility.

    Returns:
        List of loan parameter dicts ready for ContractAttributes.
    """
    rng = random.Random(seed)
    loans = []

    for i in range(n_loans):
        # Random origination: 2020-01 to 2024-06
        orig_year = rng.randint(2020, 2024)
        orig_month = rng.randint(1, 12)
        origination = ActusDateTime(orig_year, orig_month, 15)

        # Random maturity: 3–10 years from origination
        term_years = rng.randint(3, 10)
        maturity = ActusDateTime(orig_year + term_years, orig_month, 15)

        # Skip loans that already matured before valuation date
        if maturity <= valuation_date:
            continue

        rate = round(rng.uniform(0.03, 0.08), 4)
        notional = round(rng.uniform(50_000, 500_000), -3)  # round to nearest 1k
        cycle = rng.choice(["1M", "3M", "6M"])

        loans.append(
            {
                "contract_id": f"LOAN-{i:06d}",
                "origination_date": origination,
                "maturity_date": maturity,
                "nominal_interest_rate": rate,
                "notional_principal": notional,
                "interest_payment_cycle": cycle,
            }
        )

    return loans


# ---------------------------------------------------------------------------
# 2. Single-contract simulation + PV
# ---------------------------------------------------------------------------

# Day count convention used across all contracts
DCC = DayCountConvention.A360


def simulate_and_value(
    loan: dict,
    valuation_date: ActusDateTime,
    discount_rate: float,
) -> dict:
    """Simulate one PAM loan and compute its present value.

    Sets status_date = valuation_date so only future events are generated.
    Then discounts the resulting cash flows using the vectorized PV function.

    Args:
        loan: Loan parameter dict from generate_random_portfolio.
        valuation_date: Date to value the portfolio as of.
        discount_rate: Annual discount rate for PV calculation.

    Returns:
        Dict with contract_id, pv, num_events, and simulation_ms.
    """
    t0 = time.perf_counter()

    attrs = ContractAttributes(
        contract_id=loan["contract_id"],
        contract_type=ContractType.PAM,
        contract_role=ContractRole.RPA,  # Lender perspective (positive inflows)
        status_date=valuation_date,  # Only future events
        initial_exchange_date=loan["origination_date"],
        maturity_date=loan["maturity_date"],
        notional_principal=loan["notional_principal"],
        nominal_interest_rate=loan["nominal_interest_rate"],
        day_count_convention=DCC,
        interest_payment_cycle=loan["interest_payment_cycle"],
    )

    rf_obs = ConstantRiskFactorObserver(constant_value=0.0)
    contract = create_contract(attrs, rf_obs)
    result = contract.simulate()

    # Extract cashflows and year fractions for vectorized PV
    cashflows = []
    year_fracs = []
    for event in result.events:
        payoff = float(event.payoff)
        if payoff != 0.0:
            cashflows.append(payoff)
            yf = year_fraction(valuation_date, event.event_time, DCC)
            year_fracs.append(yf)

    if cashflows:
        cf_array = jnp.array(cashflows)
        yf_array = jnp.array(year_fracs)
        pv = float(present_value_vectorized(cf_array, yf_array, discount_rate))
    else:
        pv = 0.0

    elapsed_ms = (time.perf_counter() - t0) * 1000

    return {
        "contract_id": loan["contract_id"],
        "pv": pv,
        "num_events": len(result.events),
        "simulation_ms": elapsed_ms,
    }


# ---------------------------------------------------------------------------
# 3. Worker function for multiprocessing
# ---------------------------------------------------------------------------


def _worker_batch(args: tuple) -> list[dict]:
    """Process a batch of loans in a worker process.

    This function is the target for ProcessPoolExecutor. Each worker
    receives a chunk of loans and processes them sequentially.

    Args:
        args: Tuple of (loans_chunk, valuation_date, discount_rate).

    Returns:
        List of result dicts from simulate_and_value.
    """
    loans_chunk, val_date, disc_rate = args
    return [simulate_and_value(loan, val_date, disc_rate) for loan in loans_chunk]


def chunk_list(lst: list, n_chunks: int) -> list[list]:
    """Split a list into roughly equal chunks."""
    k, m = divmod(len(lst), n_chunks)
    return [lst[i * k + min(i, m) : (i + 1) * k + min(i + 1, m)] for i in range(n_chunks)]


# ---------------------------------------------------------------------------
# 4. Main: run the portfolio valuation
# ---------------------------------------------------------------------------


def main():
    print("=" * 80)
    print("  Portfolio Valuation Example — PAM Loans with status_date Optimization")
    print("=" * 80)

    # -- Configuration -------------------------------------------------------
    N_LOANS = 1_000  # Increase this for stress testing (e.g. 10_000, 100_000)
    DISCOUNT_RATE = 0.045
    VALUATION_DATE = ActusDateTime(2025, 6, 1)
    N_WORKERS = 4  # Number of parallel worker processes

    print(f"\nConfiguration:")
    print(f"  Loans:           {N_LOANS:,}")
    print(f"  Valuation date:  {VALUATION_DATE.to_iso()[:10]}")
    print(f"  Discount rate:   {DISCOUNT_RATE:.2%}")
    print(f"  Workers:         {N_WORKERS}")

    # -- Step 1: Generate portfolio ------------------------------------------
    print(f"\n{'─' * 80}")
    print("Step 1: Generating synthetic portfolio...")
    t0 = time.perf_counter()
    portfolio = generate_random_portfolio(N_LOANS, VALUATION_DATE)
    gen_ms = (time.perf_counter() - t0) * 1000
    print(f"  Generated {len(portfolio):,} active loans in {gen_ms:.1f}ms")
    print(f"  (Loans already matured before valuation date were excluded)")

    # Portfolio summary
    total_notional = sum(l["notional_principal"] for l in portfolio)
    avg_rate = sum(l["nominal_interest_rate"] for l in portfolio) / len(portfolio)
    print(f"\n  Portfolio summary:")
    print(f"    Total notional:  ${total_notional:>15,.0f}")
    print(f"    Average rate:    {avg_rate:.2%}")
    print(f"    Rate range:      {min(l['nominal_interest_rate'] for l in portfolio):.2%}"
          f" – {max(l['nominal_interest_rate'] for l in portfolio):.2%}")

    # -- Step 2: Sequential valuation ----------------------------------------
    print(f"\n{'─' * 80}")
    print("Step 2: Sequential valuation (single process)...")
    t0 = time.perf_counter()
    sequential_results = [
        simulate_and_value(loan, VALUATION_DATE, DISCOUNT_RATE) for loan in portfolio
    ]
    seq_elapsed = time.perf_counter() - t0

    total_pv_seq = sum(r["pv"] for r in sequential_results)
    total_events = sum(r["num_events"] for r in sequential_results)
    avg_ms = sum(r["simulation_ms"] for r in sequential_results) / len(sequential_results)

    print(f"  Completed in {seq_elapsed:.2f}s")
    print(f"  Total PV:       ${total_pv_seq:>15,.2f}")
    print(f"  Total events:   {total_events:,}")
    print(f"  Avg per loan:   {avg_ms:.2f}ms ({total_events / len(portfolio):.0f} events)")
    print(f"  Throughput:     {len(portfolio) / seq_elapsed:,.0f} loans/sec")

    # -- Step 3: Parallel valuation ------------------------------------------
    print(f"\n{'─' * 80}")
    print(f"Step 3: Parallel valuation ({N_WORKERS} workers)...")
    chunks = chunk_list(portfolio, N_WORKERS)
    t0 = time.perf_counter()

    # Use "spawn" context to avoid fork() + JAX multithreading conflict
    ctx = multiprocessing.get_context("spawn")
    parallel_results = []
    with ProcessPoolExecutor(max_workers=N_WORKERS, mp_context=ctx) as executor:
        futures = [
            executor.submit(_worker_batch, (chunk, VALUATION_DATE, DISCOUNT_RATE))
            for chunk in chunks
        ]
        for future in as_completed(futures):
            parallel_results.extend(future.result())

    par_elapsed = time.perf_counter() - t0
    total_pv_par = sum(r["pv"] for r in parallel_results)

    print(f"  Completed in {par_elapsed:.2f}s")
    print(f"  Total PV:       ${total_pv_par:>15,.2f}")
    print(f"  Throughput:     {len(portfolio) / par_elapsed:,.0f} loans/sec")
    print(f"  Speedup:        {seq_elapsed / par_elapsed:.1f}x vs sequential")

    # -- Step 4: Demonstrate status_date optimization ------------------------
    print(f"\n{'─' * 80}")
    print("Step 4: Demonstrating status_date optimization...")
    print("  Comparing event counts: origination-date SD vs valuation-date SD")

    # Pick a sample loan for comparison
    sample = portfolio[0]
    print(f"\n  Sample loan: {sample['contract_id']}")
    print(f"    Originated:  {sample['origination_date'].to_iso()[:10]}")
    print(f"    Matures:     {sample['maturity_date'].to_iso()[:10]}")
    print(f"    Rate:        {sample['nominal_interest_rate']:.2%}")
    print(f"    Cycle:       {sample['interest_payment_cycle']}")

    # Simulate from origination date
    attrs_full = ContractAttributes(
        contract_id=sample["contract_id"],
        contract_type=ContractType.PAM,
        contract_role=ContractRole.RPA,
        status_date=sample["origination_date"],  # From origination
        initial_exchange_date=sample["origination_date"],
        maturity_date=sample["maturity_date"],
        notional_principal=sample["notional_principal"],
        nominal_interest_rate=sample["nominal_interest_rate"],
        day_count_convention=DCC,
        interest_payment_cycle=sample["interest_payment_cycle"],
    )
    rf_obs = ConstantRiskFactorObserver(constant_value=0.0)
    t0 = time.perf_counter()
    result_full = create_contract(attrs_full, rf_obs).simulate()
    ms_full = (time.perf_counter() - t0) * 1000

    # Simulate from valuation date
    attrs_val = ContractAttributes(
        contract_id=sample["contract_id"],
        contract_type=ContractType.PAM,
        contract_role=ContractRole.RPA,
        status_date=VALUATION_DATE,  # From valuation date
        initial_exchange_date=sample["origination_date"],
        maturity_date=sample["maturity_date"],
        notional_principal=sample["notional_principal"],
        nominal_interest_rate=sample["nominal_interest_rate"],
        day_count_convention=DCC,
        interest_payment_cycle=sample["interest_payment_cycle"],
    )
    t0 = time.perf_counter()
    result_val = create_contract(attrs_val, rf_obs).simulate()
    ms_val = (time.perf_counter() - t0) * 1000

    print(f"\n  From origination: {len(result_full.events):>4} events  ({ms_full:.2f}ms)")
    print(f"  From valuation:   {len(result_val.events):>4} events  ({ms_val:.2f}ms)")
    print(f"  Events saved:     {len(result_full.events) - len(result_val.events):>4}"
          f" ({(1 - len(result_val.events) / len(result_full.events)) * 100:.0f}% reduction)")

    # -- Step 5: Batched JAX PV vs scalar PV ---------------------------------
    print(f"\n{'─' * 80}")
    print("Step 5: JAX vectorized PV vs scalar PV comparison...")

    # Collect all cashflows from the first 100 loans
    n_compare = min(100, len(portfolio))
    all_cfs = []
    all_yfs = []
    for loan in portfolio[:n_compare]:
        attrs = ContractAttributes(
            contract_id=loan["contract_id"],
            contract_type=ContractType.PAM,
            contract_role=ContractRole.RPA,
            status_date=VALUATION_DATE,
            initial_exchange_date=loan["origination_date"],
            maturity_date=loan["maturity_date"],
            notional_principal=loan["notional_principal"],
            nominal_interest_rate=loan["nominal_interest_rate"],
            day_count_convention=DCC,
            interest_payment_cycle=loan["interest_payment_cycle"],
        )
        result = create_contract(attrs, rf_obs).simulate()
        for event in result.events:
            payoff = float(event.payoff)
            if payoff != 0.0:
                all_cfs.append(payoff)
                all_yfs.append(year_fraction(VALUATION_DATE, event.event_time, DCC))

    print(f"  Total non-zero cashflows across {n_compare} loans: {len(all_cfs):,}")

    # Scalar PV (Python loop)
    t0 = time.perf_counter()
    pv_scalar = 0.0
    for cf, yf in zip(all_cfs, all_yfs):
        pv_scalar += cf / (1.0 + DISCOUNT_RATE * yf)
    scalar_ms = (time.perf_counter() - t0) * 1000

    # Vectorized PV (JAX)
    cf_array = jnp.array(all_cfs)
    yf_array = jnp.array(all_yfs)

    # Warm up JIT
    _ = present_value_vectorized(cf_array, yf_array, DISCOUNT_RATE)

    t0 = time.perf_counter()
    pv_jax = present_value_vectorized(cf_array, yf_array, DISCOUNT_RATE)
    pv_jax.block_until_ready()  # Force JAX to complete before timing
    jax_ms = (time.perf_counter() - t0) * 1000

    print(f"\n  Scalar PV (Python loop): ${pv_scalar:>15,.2f}  ({scalar_ms:.3f}ms)")
    print(f"  Vectorized PV (JAX):     ${float(pv_jax):>15,.2f}  ({jax_ms:.3f}ms)")
    print(f"  Match: {abs(pv_scalar - float(pv_jax)) < 1.0}")

    # -- Summary -------------------------------------------------------------
    print(f"\n{'─' * 80}")
    print("Summary & Performance Notes")
    print(f"{'─' * 80}")
    print(f"""
  Portfolio: {len(portfolio):,} active PAM loans
  Total PV:  ${total_pv_seq:,.2f}

  Optimizations applied:
    1. status_date = valuation_date  — skips all historical events
    2. present_value_vectorized      — JIT-compiled batch discounting
    3. ProcessPoolExecutor           — parallel simulation across {N_WORKERS} cores
    4. block_until_ready()           — accurate JAX benchmarking

  Current bottleneck:
    The per-contract simulation (event schedule generation + POF/STF loop)
    runs as Python-level code. JAX's jit/vmap cannot accelerate it because
    the loop uses Python lists, conditionals, and dataclass state objects.

  To unlock full JAX vectorization for millions of loans, the simulation
  engine would need to be rewritten as a pure JAX function operating on
  padded arrays of contract parameters — enabling jax.vmap over the entire
  portfolio in a single kernel launch. This is a substantial architectural
  change that would trade flexibility for raw throughput.

  For now, the combination of status_date filtering + multiprocessing gives
  the best practical performance for large portfolios.
""")


if __name__ == "__main__":
    main()
