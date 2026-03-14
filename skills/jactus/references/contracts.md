# ACTUS Contract Types Reference

Complete reference for all 18 ACTUS contract types supported by JACTUS. Each entry
includes the full name, description, required fields (beyond universal required fields),
commonly used optional fields, and a minimal working Python example.

## Universal Required Fields

These fields are required (or strongly recommended) for ALL contract types:

| Field | Type | Description |
|---|---|---|
| `contract_id` | str | Unique contract identifier |
| `contract_type` | ContractType | ACTUS contract type enum |
| `contract_role` | ContractRole | RPA, RPL, RFL, PFL, BUY, SEL, LG, ST |
| `status_date` | ActusDateTime | Analysis/valuation date |

---

## Principal Instruments

### PAM — Principal at Maturity

Bond or bullet loan where the entire principal is repaid at maturity. Interest is
paid periodically; principal is returned in one lump sum at the end.

**Additional Required Fields:**

| Field | Type | Description |
|---|---|---|
| `initial_exchange_date` | ActusDateTime | Contract inception date |
| `maturity_date` | ActusDateTime | Contract maturity date |
| `notional_principal` | float | Principal amount |
| `nominal_interest_rate` | float | Interest rate (decimal) |

**Common Optional Fields:** `interest_payment_cycle`, `day_count_convention`,
`currency`, `rate_reset_cycle`, `rate_reset_market_object`

```python
from jactus.contracts import create_contract
from jactus.core import ContractAttributes, ContractType, ContractRole, ActusDateTime
from jactus.observers import ConstantRiskFactorObserver

attrs = ContractAttributes(
    contract_id="PAM-001", contract_type=ContractType.PAM,
    contract_role=ContractRole.RPA,
    status_date=ActusDateTime(2024, 1, 1),
    initial_exchange_date=ActusDateTime(2024, 1, 15),
    maturity_date=ActusDateTime(2025, 1, 15),
    notional_principal=100_000.0,
    nominal_interest_rate=0.05,
)
rf = ConstantRiskFactorObserver(constant_value=0.0)
result = create_contract(attrs, rf).simulate()
```

---

### LAM — Linear Amortizer

Loan with equal principal repayments (linear amortization). The outstanding balance
decreases linearly over the life of the contract.

**Additional Required Fields:**

| Field | Type | Description |
|---|---|---|
| `initial_exchange_date` | ActusDateTime | Contract inception |
| `notional_principal` | float | Principal amount |
| `nominal_interest_rate` | float | Interest rate |
| `maturity_date` | ActusDateTime | Required if no `principal_redemption_cycle` |
| `principal_redemption_cycle` | str | Required if no `maturity_date` (e.g., "1Y") |

**Common Optional Fields:** `next_principal_redemption_amount`, `interest_payment_cycle`,
`day_count_convention`

```python
attrs = ContractAttributes(
    contract_id="LAM-001", contract_type=ContractType.LAM,
    contract_role=ContractRole.RPA,
    status_date=ActusDateTime(2024, 1, 1),
    initial_exchange_date=ActusDateTime(2024, 1, 15),
    maturity_date=ActusDateTime(2029, 1, 15),
    notional_principal=500_000.0,
    nominal_interest_rate=0.04,
    principal_redemption_cycle="1Y",
    next_principal_redemption_amount=100_000.0,
)
```

---

### LAX — Exotic Linear Amortizer

Linear amortizer with non-standard (exotic) amortization schedules defined via arrays.
Allows custom principal repayment amounts, cycles, and anchors per schedule segment.

**Additional Required Fields:**

| Field | Type | Description |
|---|---|---|
| `initial_exchange_date` | ActusDateTime | Contract inception |
| `maturity_date` | ActusDateTime | Contract maturity |
| `notional_principal` | float | Principal amount |
| `nominal_interest_rate` | float | Interest rate |
| `array_pr_cycle` | list[str] | Array of PR cycles per segment |
| `array_pr_next` | list[float] | Array of PR amounts per segment |
| `array_pr_anchor` | list[ActusDateTime] | Array of PR anchor dates |
| `array_increase_decrease` | list[str] | "INC" or "DEC" per segment |

```python
attrs = ContractAttributes(
    contract_id="LAX-001", contract_type=ContractType.LAX,
    contract_role=ContractRole.RPA,
    status_date=ActusDateTime(2024, 1, 1),
    initial_exchange_date=ActusDateTime(2024, 1, 15),
    maturity_date=ActusDateTime(2027, 1, 15),
    notional_principal=300_000.0,
    nominal_interest_rate=0.045,
    array_pr_cycle=["6M", "1Y"],
    array_pr_next=[25_000.0, 50_000.0],
    array_pr_anchor=[ActusDateTime(2024, 7, 15), ActusDateTime(2025, 1, 15)],
    array_increase_decrease=["DEC", "DEC"],
)
```

---

### NAM — Negative Amortizer

Loan where the principal redemption amount is fixed, but if interest exceeds the
payment, the excess is capitalized (negative amortization can occur).

**Additional Required Fields:**

| Field | Type | Description |
|---|---|---|
| `initial_exchange_date` | ActusDateTime | Contract inception |
| `notional_principal` | float | Principal amount |
| `nominal_interest_rate` | float | Interest rate |
| `maturity_date` | ActusDateTime | Required if no `principal_redemption_cycle` |
| `principal_redemption_cycle` | str | Required if no `maturity_date` |

**Common Optional Fields:** `next_principal_redemption_amount`

```python
attrs = ContractAttributes(
    contract_id="NAM-001", contract_type=ContractType.NAM,
    contract_role=ContractRole.RPA,
    status_date=ActusDateTime(2024, 1, 1),
    initial_exchange_date=ActusDateTime(2024, 1, 15),
    maturity_date=ActusDateTime(2029, 1, 15),
    notional_principal=500_000.0,
    nominal_interest_rate=0.06,
    principal_redemption_cycle="1M",
    next_principal_redemption_amount=5_000.0,
)
```

---

### ANN — Annuity

Standard mortgage/annuity loan with equal total payments (principal + interest)
each period. The most common consumer loan type.

**Additional Required Fields:**

| Field | Type | Description |
|---|---|---|
| `initial_exchange_date` | ActusDateTime | Contract inception |
| `notional_principal` | float | Principal amount |
| `nominal_interest_rate` | float | Interest rate |
| `principal_redemption_cycle` | str | Payment frequency (e.g., "1M") |
| `maturity_date` | ActusDateTime | Contract maturity |

**Common Optional Fields:** `amortization_date`, `day_count_convention`

```python
attrs = ContractAttributes(
    contract_id="ANN-001", contract_type=ContractType.ANN,
    contract_role=ContractRole.RPA,
    status_date=ActusDateTime(2024, 1, 1),
    initial_exchange_date=ActusDateTime(2024, 1, 15),
    maturity_date=ActusDateTime(2054, 1, 15),
    notional_principal=400_000.0,
    nominal_interest_rate=0.045,
    principal_redemption_cycle="1M",
)
```

---

### CLM — Call Money

Open-ended loan with no fixed maturity. Pays interest periodically with no
scheduled principal repayment. Either party can call the loan.

**Additional Required Fields:**

| Field | Type | Description |
|---|---|---|
| `initial_exchange_date` | ActusDateTime | Contract inception |
| `notional_principal` | float | Principal amount |
| `nominal_interest_rate` | float | Interest rate |
| `interest_payment_cycle` | str | Interest payment frequency |

Note: `maturity_date` is optional for CLM.

```python
attrs = ContractAttributes(
    contract_id="CLM-001", contract_type=ContractType.CLM,
    contract_role=ContractRole.RPA,
    status_date=ActusDateTime(2024, 1, 1),
    initial_exchange_date=ActusDateTime(2024, 1, 15),
    notional_principal=200_000.0,
    nominal_interest_rate=0.05,
    interest_payment_cycle="3M",
)
```

---

## Non-Principal Instruments

### UMP — Undefined Maturity Profile

Deposit account with no fixed maturity. Balance changes via ad-hoc transactions
(deposits and withdrawals). Requires behavioral observers for realistic modeling.

**Additional Required Fields:**

| Field | Type | Description |
|---|---|---|
| `initial_exchange_date` | ActusDateTime | Account opening date |
| `notional_principal` | float | Initial balance |
| `nominal_interest_rate` | float | Interest rate on balance |

```python
attrs = ContractAttributes(
    contract_id="UMP-001", contract_type=ContractType.UMP,
    contract_role=ContractRole.RPA,
    status_date=ActusDateTime(2024, 1, 1),
    initial_exchange_date=ActusDateTime(2024, 1, 15),
    notional_principal=50_000.0,
    nominal_interest_rate=0.02,
)
```

---

### CSH — Cash

Simplest possible contract representing a cash position. Generates a single
event at the status date.

**Additional Required Fields:**

| Field | Type | Description |
|---|---|---|
| `notional_principal` | float | Cash amount |

```python
attrs = ContractAttributes(
    contract_id="CSH-001", contract_type=ContractType.CSH,
    contract_role=ContractRole.RPA,
    status_date=ActusDateTime(2024, 1, 1),
    notional_principal=1_000_000.0,
)
```

---

### STK — Stock

Equity position representing shares of stock. Generates dividend payment events
at specified intervals.

**Additional Required Fields:**

| Field | Type | Description |
|---|---|---|
| `initial_exchange_date` | ActusDateTime | Purchase date |
| `notional_principal` | float | Position value or number of shares |

**Common Optional Fields:** `market_object_code`, `dividend_cycle`, `dividend_anchor`

```python
attrs = ContractAttributes(
    contract_id="STK-001", contract_type=ContractType.STK,
    contract_role=ContractRole.RPA,
    status_date=ActusDateTime(2024, 1, 1),
    initial_exchange_date=ActusDateTime(2024, 1, 15),
    notional_principal=10_000.0,
    market_object_code="AAPL",
    dividend_cycle="3M",
    dividend_anchor=ActusDateTime(2024, 3, 15),
)
```

---

## Exotic Instruments

### COM — Commodity

Physical commodity position. Tracks commodity value with optional quantity and
unit specifications.

**Additional Required Fields:**

| Field | Type | Description |
|---|---|---|
| `initial_exchange_date` | ActusDateTime | Purchase date |
| `notional_principal` | float | Commodity value |

**Common Optional Fields:** `quantity`, `unit`, `market_object_code`

```python
attrs = ContractAttributes(
    contract_id="COM-001", contract_type=ContractType.COM,
    contract_role=ContractRole.RPA,
    status_date=ActusDateTime(2024, 1, 1),
    initial_exchange_date=ActusDateTime(2024, 1, 15),
    notional_principal=75_000.0,
    quantity=100.0,
    unit="barrel",
    market_object_code="WTI-OIL",
)
```

---

## Derivative Instruments

### FXOUT — Foreign Exchange Outright

FX forward contract for exchanging two currency amounts at a future settlement date.

**Additional Required Fields:**

| Field | Type | Description |
|---|---|---|
| `initial_exchange_date` | ActusDateTime | Trade date |
| `maturity_date` | ActusDateTime | Settlement date |
| `notional_principal` | float | Amount in currency 1 |
| `currency` | str | ISO code for currency 1 |
| `currency_2` | str | ISO code for currency 2 |
| `notional_principal_2` | float | Amount in currency 2 |
| `delivery_settlement` | str | "D" (delivery/net) or "S" (settlement/gross) |

```python
attrs = ContractAttributes(
    contract_id="FXOUT-001", contract_type=ContractType.FXOUT,
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

---

### OPTNS — Option

Financial option on an underlying contract. Supports call, put, and collar types
with European, American, or Bermudan exercise styles.

**Additional Required Fields:**

| Field | Type | Description |
|---|---|---|
| `initial_exchange_date` | ActusDateTime | Trade date |
| `maturity_date` | ActusDateTime | Expiration date |
| `notional_principal` | float | Option notional |
| `contract_structure` | str (JSON) | Reference to underlier: `{"Underlier": "ID"}` |
| `option_type` | str | "C" (call), "P" (put), "CP" (collar) |
| `option_strike_1` | float | Primary strike price |
| `option_exercise_type` | str | "E" (European), "A" (American), "B" (Bermudan) |

```python
attrs = ContractAttributes(
    contract_id="OPTNS-001", contract_type=ContractType.OPTNS,
    contract_role=ContractRole.BUY,
    status_date=ActusDateTime(2024, 1, 1),
    initial_exchange_date=ActusDateTime(2024, 1, 15),
    maturity_date=ActusDateTime(2024, 7, 15),
    notional_principal=100_000.0,
    contract_structure='{"Underlier": "STK-001"}',
    option_type="C", option_strike_1=150.0, option_exercise_type="E",
)
```

---

### FUTUR — Future

Standardized futures contract on an underlying asset. Settles at a future date.

**Additional Required Fields:**

| Field | Type | Description |
|---|---|---|
| `initial_exchange_date` | ActusDateTime | Trade date |
| `maturity_date` | ActusDateTime | Delivery date |
| `notional_principal` | float | Contract value |
| `contract_structure` | str (JSON) | Reference to underlier: `{"Underlier": "ID"}` |

**Common Optional Fields:** `future_price`

```python
attrs = ContractAttributes(
    contract_id="FUTUR-001", contract_type=ContractType.FUTUR,
    contract_role=ContractRole.LG,
    status_date=ActusDateTime(2024, 1, 1),
    initial_exchange_date=ActusDateTime(2024, 1, 15),
    maturity_date=ActusDateTime(2024, 6, 15),
    notional_principal=50_000.0,
    contract_structure='{"Underlier": "COM-001"}',
    future_price=750.0,
)
```

---

### SWPPV — Plain Vanilla Interest Rate Swap

Single-contract interest rate swap with fixed and floating legs modeled together.
No child contracts needed (unlike SWAPS).

**Additional Required Fields:**

| Field | Type | Description |
|---|---|---|
| `initial_exchange_date` | ActusDateTime | Effective date |
| `maturity_date` | ActusDateTime | Termination date |
| `notional_principal` | float | Swap notional |
| `nominal_interest_rate` | float | Fixed leg rate |
| `nominal_interest_rate_2` | float | Initial floating rate |
| `interest_payment_cycle` | str | Payment frequency (e.g., "6M") |
| `rate_reset_cycle` | str | Floating leg reset frequency |

**Common Optional Fields:** `rate_reset_market_object`, `rate_reset_spread`,
`rate_reset_floor`, `rate_reset_cap`

```python
attrs = ContractAttributes(
    contract_id="SWPPV-001", contract_type=ContractType.SWPPV,
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

---

### SWAPS — Swap (Composite)

Composite swap contract built from two child leg contracts. Each leg is an independent
contract (typically PAM) that is simulated first, then results are fed to the parent.

**Additional Required Fields:**

| Field | Type | Description |
|---|---|---|
| `initial_exchange_date` | ActusDateTime | Effective date |
| `maturity_date` | ActusDateTime | Termination date |
| `notional_principal` | float | Swap notional |
| `contract_structure` | str (JSON) | `{"FirstLeg": "LEG1", "SecondLeg": "LEG2"}` |

**Requires child contracts:** YES — simulate two leg contracts first, register them
with `SimulatedChildContractObserver`, then pass to `create_contract`.

```python
from jactus.observers.child_contract import SimulatedChildContractObserver

rf = ConstantRiskFactorObserver(constant_value=0.05)

leg1_attrs = ContractAttributes(
    contract_id="LEG1", contract_type=ContractType.PAM,
    contract_role=ContractRole.RPA,
    status_date=ActusDateTime(2024, 1, 1),
    initial_exchange_date=ActusDateTime(2024, 1, 1),
    maturity_date=ActusDateTime(2029, 1, 1),
    notional_principal=1_000_000.0, nominal_interest_rate=0.04,
    interest_payment_cycle="6M",
)
leg2_attrs = ContractAttributes(
    contract_id="LEG2", contract_type=ContractType.PAM,
    contract_role=ContractRole.RPL,
    status_date=ActusDateTime(2024, 1, 1),
    initial_exchange_date=ActusDateTime(2024, 1, 1),
    maturity_date=ActusDateTime(2029, 1, 1),
    notional_principal=1_000_000.0, nominal_interest_rate=0.035,
    interest_payment_cycle="6M",
)

leg1_result = create_contract(leg1_attrs, rf).simulate()
leg2_result = create_contract(leg2_attrs, rf).simulate()

child_obs = SimulatedChildContractObserver()
child_obs.register_simulation("LEG1", leg1_result.events, leg1_attrs, leg1_result.initial_state)
child_obs.register_simulation("LEG2", leg2_result.events, leg2_attrs, leg2_result.initial_state)

swap_attrs = ContractAttributes(
    contract_id="SWAP-001", contract_type=ContractType.SWAPS,
    contract_role=ContractRole.RFL,
    status_date=ActusDateTime(2024, 1, 1),
    maturity_date=ActusDateTime(2029, 1, 1),
    contract_structure='{"FirstLeg": "LEG1", "SecondLeg": "LEG2"}',
)
swap = create_contract(swap_attrs, rf, child_obs)
result = swap.simulate()
```

---

### CAPFL — Cap/Floor

Interest rate cap or floor on an underlying floating-rate contract. Pays the
difference when the rate exceeds the cap (or falls below the floor).

**Additional Required Fields:**

| Field | Type | Description |
|---|---|---|
| `initial_exchange_date` | ActusDateTime | Effective date |
| `maturity_date` | ActusDateTime | Termination date |
| `notional_principal` | float | Notional |
| `rate_reset_cycle` | str | Reset frequency |

**Common Optional Fields:** `rate_reset_cap`, `rate_reset_floor`, `nominal_interest_rate`

**Requires child contracts:** YES — simulate the underlying floating-rate contract first.

```python
rf = ConstantRiskFactorObserver(constant_value=0.06)

loan_attrs = ContractAttributes(
    contract_id="LOAN-001", contract_type=ContractType.PAM,
    contract_role=ContractRole.RPA,
    status_date=ActusDateTime(2024, 1, 1),
    initial_exchange_date=ActusDateTime(2024, 1, 1),
    maturity_date=ActusDateTime(2029, 1, 1),
    notional_principal=1_000_000.0, nominal_interest_rate=0.05,
    rate_reset_cycle="3M", rate_reset_market_object="LIBOR-3M",
)
loan_result = create_contract(loan_attrs, rf).simulate()

child_obs = SimulatedChildContractObserver()
child_obs.register_simulation("LOAN-001", loan_result.events, loan_attrs, loan_result.initial_state)

cap_attrs = ContractAttributes(
    contract_id="CAP-001", contract_type=ContractType.CAPFL,
    contract_role=ContractRole.BUY,
    status_date=ActusDateTime(2024, 1, 1),
    initial_exchange_date=ActusDateTime(2024, 1, 1),
    maturity_date=ActusDateTime(2029, 1, 1),
    notional_principal=1_000_000.0,
    rate_reset_cycle="3M", rate_reset_cap=0.055,
)
cap = create_contract(cap_attrs, rf, child_obs)
result = cap.simulate()
```

---

### CEG — Credit Enhancement Guarantee

Credit guarantee or credit default swap protecting an underlying contract against
default events.

**Additional Required Fields:**

| Field | Type | Description |
|---|---|---|
| `initial_exchange_date` | ActusDateTime | Effective date |
| `maturity_date` | ActusDateTime | Termination date |
| `notional_principal` | float | Guarantee notional |
| `contract_structure` | str (JSON) | `{"CoveredContract": "LOAN-001"}` |

**Common Optional Fields:** `coverage` (ratio, e.g., 1.0 = 100%)

**Requires child contracts:** YES

```python
rf = ConstantRiskFactorObserver(constant_value=0.0)

loan_attrs = ContractAttributes(
    contract_id="LOAN-001", contract_type=ContractType.PAM,
    contract_role=ContractRole.RPA,
    status_date=ActusDateTime(2024, 1, 1),
    initial_exchange_date=ActusDateTime(2024, 1, 1),
    maturity_date=ActusDateTime(2029, 1, 1),
    notional_principal=500_000.0, nominal_interest_rate=0.04,
)
loan_result = create_contract(loan_attrs, rf).simulate()

child_obs = SimulatedChildContractObserver()
child_obs.register_simulation("LOAN-001", loan_result.events, loan_attrs, loan_result.initial_state)

ceg_attrs = ContractAttributes(
    contract_id="CEG-001", contract_type=ContractType.CEG,
    contract_role=ContractRole.BUY,
    status_date=ActusDateTime(2024, 1, 1),
    initial_exchange_date=ActusDateTime(2024, 1, 1),
    maturity_date=ActusDateTime(2029, 1, 1),
    notional_principal=500_000.0, coverage=1.0,
    contract_structure='{"CoveredContract": "LOAN-001"}',
)
ceg = create_contract(ceg_attrs, rf, child_obs)
result = ceg.simulate()
```

---

### CEC — Credit Enhancement Collateral

Credit enhancement via collateral backing an underlying contract. Similar to CEG
but represents the collateral side rather than the guarantee.

**Additional Required Fields:**

| Field | Type | Description |
|---|---|---|
| `initial_exchange_date` | ActusDateTime | Effective date |
| `maturity_date` | ActusDateTime | Termination date |
| `notional_principal` | float | Collateral notional |
| `contract_structure` | str (JSON) | `{"CoveredContract": "LOAN-001"}` |

**Common Optional Fields:** `coverage`

**Requires child contracts:** YES (same pattern as CEG)

```python
cec_attrs = ContractAttributes(
    contract_id="CEC-001", contract_type=ContractType.CEC,
    contract_role=ContractRole.BUY,
    status_date=ActusDateTime(2024, 1, 1),
    initial_exchange_date=ActusDateTime(2024, 1, 1),
    maturity_date=ActusDateTime(2029, 1, 1),
    notional_principal=500_000.0, coverage=1.0,
    contract_structure='{"CoveredContract": "LOAN-001"}',
)
# Requires child_obs with LOAN-001 registered (same as CEG example)
cec = create_contract(cec_attrs, rf, child_obs)
result = cec.simulate()
```
