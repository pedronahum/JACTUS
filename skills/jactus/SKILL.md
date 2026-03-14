---
name: jactus
description: >
  Expertise in ACTUS financial contract modeling using the JACTUS Python/JAX library.
  Use this skill whenever the user asks to simulate loans, mortgages, bonds, swaps,
  options, FX forwards, futures, interest rate caps/floors, or any structured finance
  cash flows. Also use for JAX-based automatic differentiation risk analytics (DV01,
  delta, gamma, PV01), batch portfolio simulation, or GPU-accelerated financial
  contract modeling. Activate for any question involving ACTUS contract types: PAM,
  ANN, LAM, LAX, NAM, CLM, UMP, CSH, STK, COM, FXOUT, OPTNS, FUTUR, SWPPV, SWAPS,
  CAPFL, CEG, CEC.
install:
  - pip install git+https://github.com/pedronahum/JACTUS.git
---

# JACTUS Financial Contract Skill

JACTUS is a Python library implementing the ACTUS (Algorithmic Contract Types Unified
Standards) specification using JAX for high-performance, differentiable financial contract
modeling. It supports 18 contract types covering loans, bonds, mortgages, swaps, options,
FX forwards, futures, caps/floors, credit enhancements, and more. ACTUS standardizes the
cash flow logic of financial instruments, making contract analytics deterministic and
reproducible.

## Workflow

### 1. Identify the Contract Type

Map the user's described instrument to one of 18 ACTUS types:

| Instrument | ACTUS Type |
|---|---|
| Bullet loan / bond / zero-coupon | PAM |
| Linear amortizing loan | LAM |
| Exotic amortization schedule | LAX |
| Negative amortization loan | NAM |
| Mortgage / equal payment annuity | ANN |
| Call money / revolving facility | CLM |
| Deposit account / undefined maturity | UMP |
| Cash position | CSH |
| Stock / equity position | STK |
| Commodity position | COM |
| FX forward / FX swap | FXOUT |
| Option (call/put/collar) | OPTNS |
| Futures contract | FUTUR |
| Plain vanilla interest rate swap | SWPPV |
| Composite swap (two legs) | SWAPS |
| Interest rate cap / floor | CAPFL |
| Credit guarantee / CDS | CEG |
| Credit enhancement collateral | CEC |

If the instrument is ambiguous, ask one clarifying question about the repayment
structure or settlement type.

### 2. Build ContractAttributes

Required fields for ALL contracts:

| Field | Type | Description |
|---|---|---|
| `contract_id` | str | Unique identifier |
| `contract_type` | ContractType | One of the 18 ACTUS types |
| `contract_role` | ContractRole | RPA (lender/long) or RPL (borrower/short) |
| `status_date` | ActusDateTime | Valuation date |
| `initial_exchange_date` | ActusDateTime | Contract inception |
| `maturity_date` | ActusDateTime | Contract end |
| `notional_principal` | float | Principal amount |
| `nominal_interest_rate` | float | Interest rate (decimal) |

ContractRole values: `RPA` = lender/asset/long, `RPL` = borrower/liability/short,
`RFL`/`PFL` = swap legs, `BUY`/`SEL` = protection, `LG`/`ST` = futures long/short.

ActusDateTime format: `ActusDateTime(YYYY, MM, DD)` using integer arguments.

Minimal working example (PAM loan, $100k, 5%, 1yr):

```python
from jactus.contracts import create_contract
from jactus.core import ContractAttributes, ContractType, ContractRole, ActusDateTime
from jactus.observers import ConstantRiskFactorObserver

attrs = ContractAttributes(
    contract_id="LOAN-001",
    contract_type=ContractType.PAM,
    contract_role=ContractRole.RPA,
    status_date=ActusDateTime(2024, 1, 1),
    initial_exchange_date=ActusDateTime(2024, 1, 15),
    maturity_date=ActusDateTime(2025, 1, 15),
    notional_principal=100_000.0,
    nominal_interest_rate=0.05,
)
```

Read `references/contracts.md` for the full schema of each contract type.

### 3. Choose a RiskFactorObserver

Every simulation requires a risk factor observer, even for fixed-rate contracts:

- **ConstantRiskFactorObserver(constant_value=0.0)** — deterministic, single value for
  all risk factors. Use for fixed-rate contracts or when market rates are irrelevant.
- **DictRiskFactorObserver({"LIBOR-3M": 0.05})** — maps identifiers to fixed values.
  Use when different risk factors need different constant rates.
- **TimeSeriesRiskFactorObserver** — time-varying rates with interpolation. Use for
  floating-rate contracts with rate resets.
- **Custom observer** — subclass `BaseRiskFactorObserver` and implement `get_risk_factor()`.

Read `references/risk-factors.md` for custom observer patterns.

### 4. Simulate

```python
rf_observer = ConstantRiskFactorObserver(constant_value=0.0)
contract = create_contract(attrs, rf_observer)
result = contract.simulate()

for event in result.events:
    if event.payoff != 0:
        print(f"{event.event_time} | {event.event_type} | {event.payoff:>12,.2f}")
```

The `result` is a `SimulationHistory` with:
- `events` — list of `ContractEvent` objects
- `states` — list of `ContractState` objects
- `initial_state` / `final_state` — contract state snapshots

Always filter events where `event.payoff != 0` for display to the user.

### 5. JAX Risk Analytics

Compute DV01 (interest rate sensitivity) using JAX automatic differentiation:

```python
import jax
import jax.numpy as jnp
from jactus.contracts import create_contract
from jactus.core import ContractAttributes, ContractType, ContractRole, ActusDateTime
from jactus.observers import ConstantRiskFactorObserver

def total_cashflow(rate):
    attrs = ContractAttributes(
        contract_id="LOAN-001",
        contract_type=ContractType.PAM,
        contract_role=ContractRole.RPA,
        status_date=ActusDateTime(2024, 1, 1),
        initial_exchange_date=ActusDateTime(2024, 1, 15),
        maturity_date=ActusDateTime(2025, 1, 15),
        notional_principal=100_000.0,
        nominal_interest_rate=float(rate),
    )
    rf = ConstantRiskFactorObserver(constant_value=0.0)
    result = create_contract(attrs, rf).simulate()
    return sum(e.payoff for e in result.events)

dv01 = jax.grad(total_cashflow)(0.05)
print(f"DV01: {dv01:.4f}")
```

For batch portfolio simulation, use `jax.vmap()` over contract parameters or the
array-mode portfolio API:

```python
from jactus.contracts.portfolio import simulate_portfolio
results = simulate_portfolio(contracts_list)
```

## Common Patterns

See `assets/pam_template.py` for a PAM loan with lender/borrower perspectives.
See `assets/swap_template.py` for a SWPPV interest rate swap.
See `assets/portfolio_template.py` for batch portfolio simulation.

### PAM (Bullet Loan)

```python
attrs = ContractAttributes(
    contract_id="BOND-001", contract_type=ContractType.PAM,
    contract_role=ContractRole.RPA,
    status_date=ActusDateTime(2024, 1, 1),
    initial_exchange_date=ActusDateTime(2024, 1, 15),
    maturity_date=ActusDateTime(2029, 1, 15),
    notional_principal=1_000_000.0,
    nominal_interest_rate=0.045,
    interest_payment_cycle="6M",
)
```

### ANN (Mortgage with Amortization)

```python
attrs = ContractAttributes(
    contract_id="MORT-001", contract_type=ContractType.ANN,
    contract_role=ContractRole.RPA,
    status_date=ActusDateTime(2024, 1, 1),
    initial_exchange_date=ActusDateTime(2024, 2, 1),
    maturity_date=ActusDateTime(2054, 2, 1),
    notional_principal=400_000.0,
    nominal_interest_rate=0.065,
    principal_redemption_cycle="1M",
)
```

### SWPPV (Plain Vanilla IRS)

```python
attrs = ContractAttributes(
    contract_id="IRS-001", contract_type=ContractType.SWPPV,
    contract_role=ContractRole.RFL,
    status_date=ActusDateTime(2024, 1, 1),
    initial_exchange_date=ActusDateTime(2024, 1, 1),
    maturity_date=ActusDateTime(2029, 1, 1),
    notional_principal=10_000_000.0,
    nominal_interest_rate=0.04,
    nominal_interest_rate_2=0.035,
    interest_payment_cycle="6M",
    rate_reset_cycle="3M",
    rate_reset_market_object="LIBOR-3M",
)
```

### FXOUT (FX Forward)

```python
attrs = ContractAttributes(
    contract_id="FX-001", contract_type=ContractType.FXOUT,
    contract_role=ContractRole.RPA,
    status_date=ActusDateTime(2024, 1, 1),
    initial_exchange_date=ActusDateTime(2024, 1, 15),
    maturity_date=ActusDateTime(2024, 7, 15),
    notional_principal=1_000_000.0,
    currency="USD", currency_2="EUR",
    notional_principal_2=920_000.0,
    delivery_settlement="S",
)
```

### OPTNS (European Call Option)

```python
attrs = ContractAttributes(
    contract_id="OPT-001", contract_type=ContractType.OPTNS,
    contract_role=ContractRole.BUY,
    status_date=ActusDateTime(2024, 1, 1),
    initial_exchange_date=ActusDateTime(2024, 1, 15),
    maturity_date=ActusDateTime(2024, 7, 15),
    notional_principal=100_000.0,
    contract_structure='{"Underlier": "STK-001"}',
    option_type="C", option_strike_1=150.0, option_exercise_type="E",
)
```

## Google Workspace Integration

JACTUS pairs with the `gws` CLI (Google Workspace CLI) for end-to-end financial
workflows:

- **gws drive** — fetch term sheets and contract parameters from Google Drive
- **JACTUS simulate** — run contract simulations on fetched data
- **gws sheets** — write cash flow tables and risk metrics to Google Sheets
- **gws gmail** — send summary reports via email

Read `references/gws-integration.md` for 5 complete workflow recipes.

## Troubleshooting

- **JAX not installed** — `pip install jax jaxlib`
- **ActusDateTime parsing errors** — always use integer args: `ActusDateTime(2024, 1, 1)`,
  not strings
- **Simulation returns empty events** — check that `status_date <= initial_exchange_date`
- **GPU not found** — JAX falls back to CPU silently, no action needed
- **Import errors** — `pip install git+https://github.com/pedronahum/JACTUS.git`
- **Missing risk factor observer** — every `create_contract()` call requires an observer
- **Composite contracts fail** — SWAPS, CAPFL, CEG, CEC require a `ChildContractObserver`;
  simulate child contracts first, then register them

## References

- Read `references/contracts.md` for the full schema of all 18 contract types
- Read `references/risk-factors.md` for custom observer patterns
- Read `references/gws-integration.md` for full gws workflow recipes
- Run `scripts/validate_and_simulate.py` to test a contract before returning it to user
- See `assets/pam_template.py`, `assets/swap_template.py`, `assets/portfolio_template.py`
