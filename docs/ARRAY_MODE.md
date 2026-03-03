# Array-Mode Simulation & Portfolio API

Array-mode is JACTUS's high-performance simulation path. Instead of Python-level loops over events, it uses JIT-compiled JAX kernels operating on batched arrays. This enables portfolio-scale simulation, automatic differentiation for risk analytics, and transparent GPU/TPU acceleration.

## When to Use Array-Mode vs Scalar

| | Scalar (`contract.simulate()`) | Array-mode |
|---|---|---|
| **Use case** | Single-contract inspection, debugging, detailed event analysis | Portfolios, scenario sweeps, gradient computation, Monte Carlo |
| **Output** | `SimulationHistory` with typed `ContractEvent` objects | JAX arrays: `payoffs` shape `(num_events,)` or `(B, T)` |
| **Performance** | ~100-500 contracts/sec | ~50,000-500,000+ contracts/sec (steady-state) |
| **Differentiation** | Not supported | `jax.grad` through the kernel |
| **GPU/TPU** | CPU only | Automatic hardware acceleration |

**Rule of thumb**: Use scalar for understanding a single contract's cash flows. Use array-mode whenever you're simulating more than a handful of contracts, or need gradients.

## Coverage

12 of 18 ACTUS contract types have dedicated array-mode kernels:

| Pattern | Types | Kernel | Description |
|---|---|---|---|
| **Stateful** | PAM, LAM, NAM, ANN, LAX, SWPPV | `jax.lax.scan` | Sequential event processing with state updates |
| **Simple** | CSH, STK, COM, FXOUT, FUTUR, OPTNS | Vectorized `jnp.where` | Direct payoff computation, no sequential dependency |

The remaining 6 types (CLM, UMP, SWAPS, CAPFL, CEG, CEC) fall back to the scalar Python path automatically when used through `simulate_portfolio()`.

---

## Architecture: Two-Phase Pipeline

Array-mode separates simulation into two phases:

```
Phase 1: Python Pre-computation          Phase 2: Pure JAX Kernel
(runs once per contract)                 (JIT-compiled, re-runnable)
────────────────────────────             ──────────────────────────────
ContractAttributes + Observer    ──►     Pure function of JAX arrays
        │                                        │
        ▼                                        ▼
  Schedule generation               lax.scan (stateful types)
  Year-fraction computation           or jnp.where (simple types)
  Risk factor pre-query              Branchless dispatch
  State initialization               Float32 throughout
        │                                        │
        ▼                                        ▼
  initial_state                      (final_state, payoffs)
  event_types     [T]                  payoffs shape (T,)
  year_fractions  [T]                  or (B, T) for batches
  rf_values       [T]
  params
```

**Why the split?** Schedule generation involves Python date arithmetic (calendar rules, end-of-month conventions, business day adjustments) that cannot be JIT-compiled. The numerical simulation — interest accrual, payoff calculation, state transitions — is a pure function of arrays that JIT compiles efficiently.

**Pre-compute once, simulate many times.** The batch preparation cost is amortized across scenario sweeps, gradient evaluations, and Monte Carlo runs. Only Phase 2 (the kernel) needs to re-run.

---

## Quick Start: Single Contract

```python
from jactus.contracts.pam_array import precompute_pam_arrays, simulate_pam_array_jit
from jactus.core import ContractAttributes, ContractType, ContractRole, ActusDateTime
from jactus.observers import ConstantRiskFactorObserver
import jax.numpy as jnp

attrs = ContractAttributes(
    contract_id="LOAN-001",
    contract_type=ContractType.PAM,
    contract_role=ContractRole.RPA,
    status_date=ActusDateTime(2024, 1, 1),
    initial_exchange_date=ActusDateTime(2024, 1, 15),
    maturity_date=ActusDateTime(2025, 1, 15),
    notional_principal=100_000.0,
    nominal_interest_rate=0.05,
    interest_payment_cycle="3M",
)
rf_obs = ConstantRiskFactorObserver(constant_value=0.0)

# Phase 1: Pre-compute (Python, runs once)
initial_state, event_types, year_fractions, rf_values, params = precompute_pam_arrays(attrs, rf_obs)

# Phase 2: Simulate (JIT-compiled, fast on repeated calls)
final_state, payoffs = simulate_pam_array_jit(initial_state, event_types, year_fractions, rf_values, params)

total_cashflow = float(jnp.sum(payoffs))
```

## Quick Start: Portfolio (Recommended)

For portfolios — especially mixed-type ones — use the unified `simulate_portfolio()` API:

```python
from jactus.contracts.portfolio import simulate_portfolio
from jactus.core import ContractAttributes, ContractType, ContractRole, ActusDateTime
from jactus.observers import ConstantRiskFactorObserver

rf_obs = ConstantRiskFactorObserver(constant_value=0.0)

contracts = [
    (pam_attrs, rf_obs),
    (lam_attrs, rf_obs),
    (csh_attrs, rf_obs),
    (optns_attrs, rf_obs),
]

result = simulate_portfolio(contracts, discount_rate=0.05)
# result["total_cashflows"]   -> jnp.array shape (4,), total cashflow per contract
# result["batch_contracts"]   -> 4  (all 4 used batch kernels)
# result["fallback_contracts"]-> 0
# result["types_used"]        -> {ContractType.PAM, ContractType.LAM, ...}
```

---

## Unified Portfolio API Reference

### `simulate_portfolio()`

```python
from jactus.contracts.portfolio import simulate_portfolio

def simulate_portfolio(
    contracts: list[tuple[ContractAttributes, RiskFactorObserver]],
    discount_rate: float | None = None,
) -> dict[str, Any]:
```

**Parameters:**

| Parameter | Type | Description |
|---|---|---|
| `contracts` | `list[tuple[ContractAttributes, RiskFactorObserver]]` | Contract types may be mixed freely |
| `discount_rate` | `float \| None` | If set, compute present values (passed to each type's portfolio function) |

**Returns** a dict with:

| Key | Type | Description |
|---|---|---|
| `total_cashflows` | `jnp.ndarray` shape `(N,)` | Sum of all event payoffs per contract, in input order |
| `num_contracts` | `int` | Total contracts in portfolio |
| `batch_contracts` | `int` | Contracts simulated via JIT kernels |
| `fallback_contracts` | `int` | Contracts simulated via scalar Python path |
| `types_used` | `set[ContractType]` | Contract types present in portfolio |
| `per_type_results` | `dict[ContractType, dict]` | Raw result from each type's batch function |

**How it works:**
1. Groups contracts by `ContractType`
2. For batch-supported types (12): dispatches to `simulate_<type>_portfolio()`
3. For fallback types (6): runs scalar `create_contract(...).simulate()` per contract
4. Reassembles results in original input order

### `BATCH_SUPPORTED_TYPES`

```python
from jactus.contracts.portfolio import BATCH_SUPPORTED_TYPES

# frozenset of: PAM, LAM, NAM, ANN, LAX, CSH, STK, COM, FXOUT, FUTUR, OPTNS, SWPPV
```

---

## Per-Type Array API

Each of the 12 array-mode contract types follows the same function pattern. The functions are importable from their respective modules:

```python
from jactus.contracts.<type>_array import (
    precompute_<type>_arrays,       # Phase 1: attrs + observer → JAX arrays
    simulate_<type>_array,          # Phase 2: single-contract kernel
    simulate_<type>_array_jit,      # JIT-compiled single-contract kernel (stateful types)
    batch_simulate_<type>,          # Batched kernel: [B, T] arrays
    batch_simulate_<type>_auto,     # Device-aware dispatch (CPU: batch, GPU: vmap)
    simulate_<type>_portfolio,      # End-to-end: list of contracts → result dict
)
```

> Not all types export every function. Simple types may omit `simulate_<type>_array_jit`. All types export `precompute`, `simulate_array`, `batch_simulate`, `batch_simulate_auto`, and `simulate_portfolio`.

### Function Signatures

**Pre-computation** (all types):
```python
def precompute_<type>_arrays(
    attrs: ContractAttributes,
    rf_observer: RiskFactorObserver,
) -> tuple[<Type>ArrayState, jnp.ndarray, jnp.ndarray, jnp.ndarray, <Type>ArrayParams]:
    """Returns (initial_state, event_types, year_fractions, rf_values, params)."""
```

**Single-contract simulation** (all types):
```python
def simulate_<type>_array(
    initial_state: <Type>ArrayState,     # NamedTuple of scalar jnp.ndarray
    event_types: jnp.ndarray,            # shape (T,), int32
    year_fractions: jnp.ndarray,         # shape (T,), float32
    rf_values: jnp.ndarray,              # shape (T,), float32
    params: <Type>ArrayParams,           # NamedTuple of scalar jnp.ndarray
) -> tuple[<Type>ArrayState, jnp.ndarray]:
    """Returns (final_state, payoffs) where payoffs is shape (T,)."""
```

**Batch simulation** (all types):
```python
def batch_simulate_<type>(
    initial_states: <Type>ArrayState,    # each field shape [B]
    event_types: jnp.ndarray,            # shape [B, T]
    year_fractions: jnp.ndarray,         # shape [B, T]
    rf_values: jnp.ndarray,              # shape [B, T]
    params: <Type>ArrayParams,           # each field shape [B]
) -> tuple[<Type>ArrayState, jnp.ndarray]:
    """JIT-compiled. Returns (final_states, payoffs) with payoffs shape [B, T]."""
```

**Portfolio simulation** (all types):
```python
def simulate_<type>_portfolio(
    contracts: list[tuple[ContractAttributes, RiskFactorObserver]],
    discount_rate: float | None = None,
) -> dict[str, Any]:
    """End-to-end: pre-compute + batch + mask.
    Returns dict with 'payoffs', 'masks', 'total_cashflows', 'final_states', etc."""
```

---

## Per-Type Reference

### Stateful Types (lax.scan)

These types track evolving contract state across events using `jax.lax.scan`. Each event may update the notional principal, accrued interest, rate, fees, or scaling factors.

#### PAM (Principal at Maturity)

- **Module**: `jactus.contracts.pam_array`
- **State** (`PAMArrayState`): `nt`, `ipnr`, `ipac`, `feac`, `nsc`, `isc` (6 fields)
- **Params** (`PAMArrayParams`): `role_sign`, `notional_principal`, `nominal_interest_rate`, `premium_discount_at_ied`, `accrued_interest`, `fee_rate`, `fee_basis`, `penalty_rate`, `penalty_type`, `price_at_purchase_date`, `price_at_termination_date`, `rate_reset_spread`, `rate_reset_multiplier`, `rate_reset_floor`, `rate_reset_cap`, `rate_reset_next`, `has_rate_floor`, `has_rate_cap`, `ied_ipac` (19 fields)
- **Key events**: IED, IP, MD, RR, RRF, PP, PY, FP, PRD, TD, SC, IPCI, AD, CE
- **Notes**: Reference implementation. Also exports `batch_precompute_pam()` for pure-JAX batch schedule generation (GPU/TPU-ready) and `prepare_pam_batch()` for batch preparation.

#### LAM (Linear Amortizer)

- **Module**: `jactus.contracts.lam_array`
- **State** (`LAMArrayState`): PAM fields + `prnxt`, `ipcb` (8 fields)
- **Params** (`LAMArrayParams`): PAM fields + `next_principal_redemption_amount`, `ipcb_mode` (0=NT, 1=NTIED, 2=NTL), `interest_calculation_base_amount`
- **Key events**: PAM events + PR, IPCB
- **Notes**: Interest calculated on `ipcb` (not `nt`) when `ipcb_mode` is NTL.

#### NAM (Negative Amortizer)

- **Module**: `jactus.contracts.nam_array`
- **State** (`NAMArrayState`): Same structure as LAM (8 fields)
- **Params** (`NAMArrayParams`): Same structure as LAM
- **Key events**: Same as LAM
- **Notes**: PR payoff differs — allows negative amortization (principal can increase).

#### ANN (Annuity)

- **Module**: `jactus.contracts.ann_array`
- **State**: Reuses `NAMArrayState` (8 fields)
- **Params**: Reuses `NAMArrayParams`
- **Key events**: Same as NAM + PRF (principal redemption fix)
- **Notes**: Reuses the NAM kernel. The difference is in pre-computation: `prnxt` is calculated using the annuity formula instead of being a fixed amount. Exports `precompute_ann_arrays`, `simulate_ann_array`, `simulate_ann_portfolio`.

#### LAX (Exotic Linear Amortizer)

- **Module**: `jactus.contracts.lax_array`
- **State** (`LAXArrayState`): Same structure as LAM (8 fields)
- **Params** (`LAXArrayParams`): Same structure as LAM
- **Key events**: LAM events + PI (principal increase)
- **Notes**: Unlike LAM/NAM, `prnxt` varies per event via a `prnxt_schedule` array. Supports PI events that increase the notional.

#### SWPPV (Plain Vanilla Interest Rate Swap)

- **Module**: `jactus.contracts.swppv_array`
- **State** (`SWPPVArrayState`): `nt`, `ipnr`, `ipac1`, `ipac2`, `nsc`, `isc` (6 fields)
- **Params** (`SWPPVArrayParams`): `role_sign`, `notional_principal`, `fixed_rate`, `rate_reset_spread`, `rate_reset_multiplier`, `rate_reset_floor`, `rate_reset_cap`, `has_rate_floor`, `has_rate_cap`, `price_at_purchase_date`, `price_at_termination_date` (11 fields)
- **Key events**: IED, IP, MD, RR, PRD, TD, AD, CE
- **Notes**: Dual-accrual model. `ipac1` = fixed leg, `ipac2` = floating leg. Net IP payoff: `role_sign * nsc * isc * ((ipac1 + yf*fixed_rate*nt) - (ipac2 + yf*ipnr*nt))`. No notional exchange at IED/MD.

### Simple Types (Vectorized)

These types have no sequential state dependency. Payoffs are computed directly from event types and static parameters using `jnp.where`. No `lax.scan` is needed.

#### CSH (Cash)

- **Module**: `jactus.contracts.csh_array`
- **State** (`CSHArrayState`): `nt` (1 field)
- **Params** (`CSHArrayParams`): `role_sign`, `notional_principal`
- **Key events**: AD
- **Notes**: Trivial — all payoffs are 0.0.

#### STK (Stock)

- **Module**: `jactus.contracts.stk_array`
- **State** (`STKArrayState`): `nt` (1 field, always 0.0)
- **Params** (`STKArrayParams`): `role_sign`, `pprd`, `ptd`
- **Key events**: PRD, TD, DV
- **Notes**: DV (dividend) payoff uses `rf_values` for the dividend amount.

#### COM (Commodity)

- **Module**: `jactus.contracts.com_array`
- **State** (`COMArrayState`): `nt` (1 field, always 0.0)
- **Params** (`COMArrayParams`): `role_sign`, `pprd`, `ptd`, `quantity`
- **Key events**: PRD, TD
- **Notes**: Payoffs multiplied by `quantity`.

#### FXOUT (FX Outright)

- **Module**: `jactus.contracts.fxout_array`
- **State** (`FXOUTArrayState`): `nt` (1 field, always 0.0)
- **Params** (`FXOUTArrayParams`): `role_sign`, `pprd`, `ptd`, `notional_1`, `notional_2`
- **Key events**: PRD, TD, MD, STD
- **Notes**: Dual-currency settlement. MD pays `notional_1`, STD pays `-notional_2`.

#### FUTUR (Futures)

- **Module**: `jactus.contracts.futur_array`
- **State** (`FUTURArrayState`): `nt` (1 field, always 0.0)
- **Params** (`FUTURArrayParams`): `role_sign`, `pprd`, `ptd`, `nt`
- **Key events**: PRD, TD, MD, XD, STD
- **Notes**: Settlement amount at XD is pre-computed from the exercise value in `rf_values`.

#### OPTNS (Options)

- **Module**: `jactus.contracts.optns_array`
- **State** (`OPTNSArrayState`): `nt` (1 field, always 0.0)
- **Params** (`OPTNSArrayParams`): `role_sign`, `pprd`, `ptd`
- **Key events**: PRD, TD, MD, XD, STD
- **Notes**: Exercise payoff at XD uses the intrinsic value from `rf_values`.

---

## Batch Processing Pipeline

When you call `simulate_<type>_portfolio()`, three steps happen internally:

### Step 1: Pre-compute (Python, per contract)

```python
# For each contract in the portfolio:
initial_state, event_types, year_fractions, rf_values, params = precompute_<type>_arrays(attrs, rf_obs)
```

Generates the event schedule, computes year fractions, pre-queries risk factors, and initializes state. This is standard Python — no JAX overhead.

### Step 2: Pad and Stack (`prepare_<type>_batch`)

```python
# Stack all contracts into [B, T] arrays:
batched_states, batched_et, batched_yf, batched_rf, batched_params, batched_masks = prepare_<type>_batch(contracts)
```

Contracts have different numbers of events. Padding aligns them to a uniform length `T = max_events` using `NOP_EVENT_IDX` (a no-op event type that produces zero payoff). The returned `masks` array is `1.0` for real events and `0.0` for padding.

### Step 3: Simulate (`batch_simulate_<type>_auto`)

```python
final_states, payoffs = batch_simulate_<type>_auto(
    batched_states, batched_et, batched_yf, batched_rf, batched_params
)
masked_payoffs = payoffs * batched_masks  # zero out padding
total_cashflows = jnp.sum(masked_payoffs, axis=1)  # shape [B]
```

Device-aware dispatch:
- **CPU**: Uses `batch_simulate_<type>` — processes all contracts together in shaped `[B, T]` arrays via a single `lax.scan` (no vmap overhead)
- **GPU/TPU**: Uses `jax.vmap(simulate_<type>_array)` — maps the single-contract kernel across the batch dimension

---

## Performance

### JIT Compilation

The first call to a batch kernel includes XLA compilation overhead (~1-5 seconds). Subsequent calls with the same array shapes reuse the compiled kernel:

```python
# First call: includes compilation
final_states, payoffs = batch_simulate_pam(states, et, yf, rf, params)
payoffs.block_until_ready()  # ~1-3s (compilation + execution)

# Second call onwards: cached kernel
final_states, payoffs = batch_simulate_pam(states, et, yf, rf, params)
payoffs.block_until_ready()  # ~0.001-0.01s (execution only)
```

To persist compiled kernels across process restarts:

```python
import jax
jax.config.update("jax_compilation_cache_dir", "/tmp/jax_cache")
```

### Float Precision

| Backend | Default dtype | float64 support |
|---|---|---|
| CPU | float32 | Yes (enable with `jax_enable_x64`) |
| GPU | float32 | Yes (enable with `jax_enable_x64`) |
| TPU | float32 | **Not supported** |

Float32 is sufficient for ACTUS cross-validation (tolerance: +/-1.0). For high-notional or long-dated instruments:

```python
import jax
jax.config.update("jax_enable_x64", True)  # call before importing JACTUS
```

### Scan Unrolling

Stateful kernels use `lax.scan(..., unroll=8)` to reduce GPU kernel launches by 8x compared to un-unrolled scans.

### Memory

Batch arrays have shape `[B, T]` where:
- `B` = number of contracts in the batch
- `T` = maximum events across all contracts in the batch (padded)

Typical `T` values: 10-200 for most contract types. Memory usage is approximately `B * T * 4 bytes * 3 arrays` (event_types, year_fractions, rf_values).

---

## Automatic Differentiation

Because the simulation kernel is a pure JAX function, `jax.grad` works through it:

```python
import jax
from jactus.contracts.pam_array import precompute_pam_arrays, simulate_pam_array

# Pre-compute (outside the gradient boundary)
initial_state, et, yf, rf, params = precompute_pam_arrays(attrs, rf_obs)

# Define PV as a function of rate
def pv_fn(rate):
    new_params = params._replace(nominal_interest_rate=rate)
    new_state = initial_state._replace(ipnr=rate)
    _, payoffs = simulate_pam_array(new_state, et, yf, rf, new_params)
    cum_yf = jnp.cumsum(yf)
    discount_factors = 1.0 / (1.0 + 0.05 * cum_yf)
    return jnp.sum(payoffs * discount_factors)

# Compute gradient: dPV/dRate
grad_fn = jax.grad(pv_fn)
rate = jnp.array(0.05)
dpv_drate = grad_fn(rate)
```

**Limitation**: The pre-computation phase is not differentiable (Python date arithmetic). Gradients flow through the Phase 2 kernel only. To vary parameters that affect the schedule (e.g., maturity date, cycle), re-run pre-computation with different attributes.

---

## GPU/TPU Acceleration

No code changes are needed — install the appropriate JAX backend:

```bash
# GPU (CUDA)
pip install "jax[cuda13]"

# TPU (Google Cloud)
pip install "jax[tpu]" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
```

JACTUS automatically detects the backend and selects the optimal execution strategy via `batch_simulate_<type>_auto`.

---

## Types Without Array-Mode

Six contract types fall back to the scalar Python path:

| Type | Reason |
|---|---|
| **CLM** (Call Money) | Dynamic event schedules — call events change the timeline at runtime |
| **UMP** (Undefined Maturity Profile) | Deposit transactions inject events dynamically |
| **SWAPS** (Generic Swap) | Composite contract requiring child contract simulation |
| **CAPFL** (Cap/Floor) | Composite contract requiring child contract simulation |
| **CEG** (Credit Enhancement Guarantee) | Composite contract requiring child contract simulation |
| **CEC** (Credit Enhancement Collateral) | Composite contract requiring child contract simulation |

These types work transparently through `simulate_portfolio()` — they are simply routed to `create_contract(...).simulate()` instead of a batch kernel.

---

## Shared Infrastructure: `array_common.py`

The `jactus.contracts.array_common` module provides shared constants, helpers, and data structures used by all array-mode implementations:

- **`NOP_EVENT_IDX`**: Padding marker (event type index beyond all valid types). Produces zero payoff in all kernels.
- **`F32`**: Alias for `jnp.float32`.
- **Event type indices**: `IED_IDX`, `IP_IDX`, `MD_IDX`, `PR_IDX`, `RR_IDX`, etc. — cached integer indices for branchless dispatch.
- **`RawPrecomputed`**: NamedTuple container for Python-level pre-computation results before JAX conversion.
- **`BatchContractParams`**: NamedTuple with schedule parameters extracted into JAX arrays for batch pre-computation.
- **`get_role_sign(role)`**: Returns `+1.0` (RPA) or `-1.0` (RPL).
- **`encode_fee_basis(attrs)`**: Encodes fee basis enum as integer for `jnp.where`.
- **`pad_arrays(et, yf, rf, max_events)`**: Pads arrays to uniform length with `NOP_EVENT_IDX`.
- **Year-fraction functions**: `yf_a360`, `yf_a365`, `yf_30e360`, `yf_b30360` (Python scalar) and `np_yf_30e360`, `np_yf_b30360` (NumPy vectorized).
- **Schedule helpers**: `fast_schedule()`, `fast_month_schedule()` — fast month-based date arithmetic for event schedule generation.

---

## Examples

- **[portfolio_valuation_vectorized_example.py](../examples/portfolio_valuation_vectorized_example.py)**: Benchmark comparing Python path vs array-mode for 500 PAM loans, including scenario analysis and gradient computation.
- **[05_gpu_tpu_portfolio_benchmark.ipynb](../examples/notebooks/05_gpu_tpu_portfolio_benchmark.ipynb)**: Interactive GPU/TPU portfolio benchmark notebook.
- **[06_gallery_of_contracts.ipynb](../examples/notebooks/06_gallery_of_contracts.ipynb)**: All 18 contract types including portfolio API examples.
