#!/usr/bin/env python3
"""
Vectorized Portfolio Valuation — Array-Mode PAM Simulation with jax.jit + jax.vmap
===================================================================================

This example compares the original Python-level simulation path with the new
array-mode path that uses ``jax.lax.scan`` for the event loop and ``jax.vmap``
for portfolio-level vectorization.

Architecture:
    Python path:   contract.simulate() → Python loop over events → per-event POF/STF
    Array path:    prepare_pam_batch() → jax.lax.scan (JIT-compiled) → jax.vmap

The array-mode pre-computes event schedules and year fractions (Python, once),
then runs the numerical simulation as a pure JAX function. This enables:
  - ``jax.jit``:  Compile the scan loop to XLA, eliminating Python overhead
  - ``jax.vmap``: Vectorize across the entire portfolio in a single kernel
  - ``jax.grad``: Automatic differentiation of portfolio PV w.r.t. any parameter

Example: 500 PAM loans, comparing throughput and PV equivalence between paths.
"""

import random
import time

import jax
import jax.numpy as jnp

from jactus.contracts import create_contract
from jactus.contracts.pam_array import (
    batch_simulate_pam,
    precompute_pam_arrays,
    prepare_pam_batch,
    simulate_pam_array,
    simulate_pam_array_jit,
)
from jactus.core import (
    ActusDateTime,
    ContractAttributes,
    ContractRole,
    ContractType,
    DayCountConvention,
)
from jactus.observers import ConstantRiskFactorObserver
from jactus.utilities import present_value_vectorized, year_fraction

DCC = DayCountConvention.A360
VALUATION_DATE = ActusDateTime(2025, 6, 1)
DISCOUNT_RATE = 0.045


# ---------------------------------------------------------------------------
# Portfolio generation (same as portfolio_valuation_example.py)
# ---------------------------------------------------------------------------


def generate_portfolio(n_loans: int, seed: int = 42) -> list[dict]:
    rng = random.Random(seed)
    loans = []
    for i in range(n_loans):
        orig_year = rng.randint(2020, 2024)
        orig_month = rng.randint(1, 12)
        origination = ActusDateTime(orig_year, orig_month, 15)
        term_years = rng.randint(3, 10)
        maturity = ActusDateTime(orig_year + term_years, orig_month, 15)
        if maturity <= VALUATION_DATE:
            continue
        rate = round(rng.uniform(0.03, 0.08), 4)
        notional = round(rng.uniform(50_000, 500_000), -3)
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


def make_attrs(loan: dict) -> ContractAttributes:
    return ContractAttributes(
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


# ---------------------------------------------------------------------------
# Python path: simulate + PV per contract
# ---------------------------------------------------------------------------


def python_path_pv(loan: dict, rf_obs) -> float:
    attrs = make_attrs(loan)
    contract = create_contract(attrs, rf_obs)
    result = contract.simulate()
    cashflows, year_fracs = [], []
    for event in result.events:
        payoff = float(event.payoff)
        if payoff != 0.0:
            cashflows.append(payoff)
            year_fracs.append(year_fraction(VALUATION_DATE, event.event_time, DCC))
    if cashflows:
        return float(
            present_value_vectorized(jnp.array(cashflows), jnp.array(year_fracs), DISCOUNT_RATE)
        )
    return 0.0


# ---------------------------------------------------------------------------
# Main benchmark
# ---------------------------------------------------------------------------


def main():
    print("=" * 80)
    print("  Array-Mode PAM Benchmark: Python path vs JIT+vmap path")
    print("=" * 80)

    n_loans = 500
    rf_obs = ConstantRiskFactorObserver(constant_value=0.0)
    portfolio = generate_portfolio(n_loans)
    n_active = len(portfolio)
    print(f"\nPortfolio: {n_active} active loans (from {n_loans} generated)")

    # -----------------------------------------------------------------------
    # 1. Python path (sequential)
    # -----------------------------------------------------------------------
    print(f"\n{'─' * 80}")
    print("1. Python path (sequential contract.simulate())")
    t0 = time.perf_counter()
    python_pvs = [python_path_pv(loan, rf_obs) for loan in portfolio]
    python_elapsed = time.perf_counter() - t0
    python_total_pv = sum(python_pvs)

    print(f"   Total PV:    ${python_total_pv:>15,.2f}")
    print(f"   Time:        {python_elapsed:.2f}s")
    print(f"   Throughput:  {n_active / python_elapsed:,.0f} loans/sec")

    # -----------------------------------------------------------------------
    # 2. Array-mode: prepare_pam_batch (pre-compute + batch stack)
    # -----------------------------------------------------------------------
    print(f"\n{'─' * 80}")
    print("2. Array-mode: prepare_pam_batch (pre-compute + batch)")
    contracts = [(make_attrs(loan), rf_obs) for loan in portfolio]

    t0 = time.perf_counter()
    batched_states, batched_et, batched_yf, batched_rf, batched_params, batched_masks = (
        prepare_pam_batch(contracts)
    )
    batch_prep_elapsed = time.perf_counter() - t0
    max_events = batched_et.shape[1]
    print(
        f"   Batch prep:    {batch_prep_elapsed:.3f}s"
        f" ({n_active / batch_prep_elapsed:,.0f} contracts/sec,"
        f" pad to {max_events} events)"
    )

    # -----------------------------------------------------------------------
    # 3. Array-mode: batched vmap simulation
    # -----------------------------------------------------------------------
    print(f"\n{'─' * 80}")
    print("3. Array-mode: vmap kernel (JIT-compiled)")

    # Warm-up JIT+vmap compilation
    t0 = time.perf_counter()
    final_states, payoffs = batch_simulate_pam(
        batched_states, batched_et, batched_yf, batched_rf, batched_params
    )
    payoffs.block_until_ready()
    vmap_warmup = time.perf_counter() - t0
    print(f"   vmap compile:  {vmap_warmup:.3f}s (first call, includes XLA compilation)")

    # Steady-state: re-run the compiled kernel (this is the real throughput)
    n_reruns = 5
    t0 = time.perf_counter()
    for _ in range(n_reruns):
        final_states, payoffs = batch_simulate_pam(
            batched_states, batched_et, batched_yf, batched_rf, batched_params
        )
        payoffs.block_until_ready()
    vmap_elapsed = (time.perf_counter() - t0) / n_reruns

    masked_payoffs = payoffs * batched_masks
    total_cf = float(jnp.sum(masked_payoffs))

    # PV with discounting
    cum_yfs = jnp.cumsum(batched_yf, axis=1)
    disc_factors = 1.0 / (1.0 + DISCOUNT_RATE * cum_yfs)
    vmap_pvs = jnp.sum(masked_payoffs * disc_factors, axis=1)
    vmap_total_pv = float(jnp.sum(vmap_pvs))

    print(f"   Steady-state:  {vmap_elapsed:.4f}s ({n_active / vmap_elapsed:,.0f} contracts/sec)")
    print(f"   Total PV:      ${vmap_total_pv:>15,.2f}")
    print(f"   Total cashflow: ${total_cf:>15,.2f}")

    # -----------------------------------------------------------------------
    # 4. Scenario analysis demo: re-run with different discount rates
    # -----------------------------------------------------------------------
    print(f"\n{'─' * 80}")
    print("4. Scenario analysis: 100 discount rate scenarios")
    print("   (Demonstrates the value of JIT — pre-compute once, run many times)")

    n_scenarios = 100
    rates = jnp.linspace(0.01, 0.10, n_scenarios)
    t0 = time.perf_counter()
    scenario_pvs = []
    for r in rates:
        disc = 1.0 / (1.0 + float(r) * cum_yfs)
        pv = float(jnp.sum(masked_payoffs * disc))
        scenario_pvs.append(pv)
    scenario_elapsed = time.perf_counter() - t0
    print(
        f"   {n_scenarios} scenarios in {scenario_elapsed:.3f}s"
        f" ({n_scenarios / scenario_elapsed:,.0f} scenarios/sec)"
    )
    print(
        f"   PV range: ${min(scenario_pvs):,.0f} (rate={float(rates[-1]):.2%})"
        f" to ${max(scenario_pvs):,.0f} (rate={float(rates[0]):.2%})"
    )

    # -----------------------------------------------------------------------
    # 5. Equivalence check
    # -----------------------------------------------------------------------
    print(f"\n{'─' * 80}")
    print("5. Equivalence check: Python path vs array-mode")

    # Compare per-contract cashflows (undiscounted) for first 20 contracts
    n_check = min(20, n_active)
    max_diff = 0.0
    for i in range(n_check):
        init_s, et, yf, rf, params = precompute_pam_arrays(make_attrs(portfolio[i]), rf_obs)
        _, check_payoffs = simulate_pam_array_jit(init_s, et, yf, rf, params)
        array_cf = float(jnp.sum(check_payoffs))

        # Python path total cashflow
        attrs = make_attrs(portfolio[i])
        contract = create_contract(attrs, rf_obs)
        py_result = contract.simulate()
        py_cf = sum(float(e.payoff) for e in py_result.events)

        diff = abs(array_cf - py_cf)
        max_diff = max(max_diff, diff)

    print(f"   Checked {n_check} contracts")
    print(f"   Max cashflow difference: ${max_diff:.6f}")
    print(f"   Within ACTUS tolerance (1.0): {'YES' if max_diff <= 1.0 else 'NO'}")

    # -----------------------------------------------------------------------
    # 6. Gradient demo: dPV/dRate
    # -----------------------------------------------------------------------
    print(f"\n{'─' * 80}")
    print("6. Gradient demo: dPV/dRate (automatic differentiation)")

    # Use first contract
    init_s, et, yf, rf, params = precompute_pam_arrays(make_attrs(portfolio[0]), rf_obs)

    def pv_as_fn_of_rate(rate):
        new_params = params._replace(nominal_interest_rate=rate)
        new_state = init_s._replace(ipnr=rate)
        _, grad_payoffs = simulate_pam_array(new_state, et, yf, rf, new_params)
        cum_yf = jnp.cumsum(yf)
        df = 1.0 / (1.0 + DISCOUNT_RATE * cum_yf)
        return jnp.sum(grad_payoffs * df)

    grad_fn = jax.grad(pv_as_fn_of_rate)
    base_rate = jnp.array(float(portfolio[0]["nominal_interest_rate"]))
    pv_val = pv_as_fn_of_rate(base_rate)
    grad_val = grad_fn(base_rate)

    print(f"   Contract: {portfolio[0]['contract_id']}")
    print(f"   Rate:     {float(base_rate):.4f}")
    print(f"   PV:       ${float(pv_val):>12,.2f}")
    print(f"   dPV/dR:   ${float(grad_val):>12,.2f}")
    print("   (A positive dPV/dR means higher rate → more interest → higher PV for lender)")

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    e2e_elapsed = batch_prep_elapsed + vmap_elapsed
    print(f"\n{'=' * 80}")
    print("Performance Summary")
    print(f"{'=' * 80}")
    print(f"""
  Portfolio: {n_active} active PAM loans

  {"Path":<40} {"Time":>10} {"Throughput":>18}
  {"─" * 70}
  Python (sequential simulate + PV)      {python_elapsed:>8.2f}s   {n_active / python_elapsed:>12,.0f} loans/sec
  Array batch prep (once)                {batch_prep_elapsed:>8.3f}s   {n_active / batch_prep_elapsed:>12,.0f} contracts/sec
  Array vmap kernel (steady-state)       {vmap_elapsed:>8.4f}s   {n_active / vmap_elapsed:>12,.0f} contracts/sec
  End-to-end (prep + kernel)             {e2e_elapsed:>8.3f}s   {n_active / e2e_elapsed:>12,.0f} contracts/sec
  Scenario sweep (100 rates)             {scenario_elapsed:>8.3f}s   {n_scenarios / scenario_elapsed:>12,.0f} scenarios/sec

  Speedup vs Python:
    Kernel only:     {python_elapsed / vmap_elapsed:>6.0f}x
    End-to-end:      {python_elapsed / e2e_elapsed:>6.0f}x  (incl. batch prep)

  Key insight: The array-mode separates the work into two phases:
    1. Batch preparation ({batch_prep_elapsed:.3f}s) — runs once per portfolio
    2. JIT kernel ({vmap_elapsed:.4f}s) — runs as many times as needed

  This makes the array-mode ideal for:
    - Scenario analysis (vary rates, run 1000s of times)
    - Gradient computation (jax.grad through the entire simulation)
    - Sensitivity sweeps (dPV/dRate, dPV/dNotional, etc.)
    - Monte Carlo simulations (re-run kernel with sampled risk factors)

  The batch preparation cost is amortized across all re-runs of the kernel.
""")


if __name__ == "__main__":
    main()
