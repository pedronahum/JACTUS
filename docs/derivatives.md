# Derivative Contracts in JACTUS

This document provides comprehensive documentation for derivative contract types implemented in JACTUS, following the ACTUS standard.

## Overview

JACTUS implements 8 derivative contract types covering foreign exchange, interest rate, and credit derivatives:

1. **FXOUT** - Foreign Exchange Outright
2. **OPTNS** - Options
3. **FUTUR** - Futures
4. **SWPPV** - Plain Vanilla Interest Rate Swap
5. **SWAPS** - Generic Swap (Composition)
6. **CAPFL** - Cap/Floor
7. **CEG** - Credit Enhancement Guarantee
8. **CEC** - Credit Enhancement Collateral

All derivative contracts are fully JAX-compatible for automatic differentiation and vectorization.

## Foreign Exchange Derivatives

### FXOUT - Foreign Exchange Outright

Foreign exchange outright contracts represent agreements to exchange currencies at a specified future date and rate.

**Key Features:**
- Dual currency exchange (primary and secondary)
- Fixed exchange rate
- Settlement at maturity
- Net or gross settlement options

**Use Cases:**
- FX hedging for known future cash flows
- FX swaps (near and far legs)
- Forward FX contracts
- Currency risk management

**Example:**
```python
from jactus.contracts import create_contract
from jactus.core import ContractAttributes, ContractType, ContractRole, ActusDateTime
from jactus.observers import ConstantRiskFactorObserver

# EUR/USD FX Outright
attrs = ContractAttributes(
    contract_id="FX-001",
    contract_type=ContractType.FXOUT,
    contract_role=ContractRole.RPA,
    status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
    maturity_date=ActusDateTime(2025, 1, 1, 0, 0, 0),
    settlement_date=ActusDateTime(2025, 1, 1, 0, 0, 0),
    currency="EUR",
    currency_2="USD",
    notional_principal=1_000_000.0,  # EUR 1M
    notional_principal_2=1_100_000.0,  # USD 1.1M
    delivery_settlement="S",  # Gross settlement
    purchase_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
    price_at_purchase_date=1.10,  # 1 EUR = 1.10 USD
)

rf_observer = ConstantRiskFactorObserver(1.12)  # Forward rate
contract = create_contract(attrs, rf_observer)
```

**For a complete working example, see:**
- `examples/fx_swap_example.py` - EUR/USD FX Swap with 1-year maturity

## Interest Rate Derivatives

### SWPPV - Plain Vanilla Interest Rate Swap

Plain vanilla swaps exchange fixed and floating interest rate payments on a notional principal.

**Key Features:**
- Fixed rate on reference leg
- Floating rate on variable leg
- Periodic interest payments
- Rate resets for floating leg
- Net settlement of interest differentials

**Use Cases:**
- Convert fixed-rate debt to floating
- Convert floating-rate debt to fixed
- Interest rate risk management
- Speculation on rate movements
- Asset-liability matching

**Example:**
```python
# 5-year interest rate swap: fixed vs O/N floating
attrs = ContractAttributes(
    contract_id="IRS-001",
    contract_type=ContractType.SWPPV,
    contract_role=ContractRole.RFL,  # Pays fixed
    status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
    maturity_date=ActusDateTime(2029, 1, 1, 0, 0, 0),
    notional_principal=10_000_000.0,
    nominal_interest_rate=0.05,  # 5% fixed
    nominal_interest_rate_2=0.03,  # 3% initial floating
    interest_payment_cycle="3M",  # Quarterly
    cycle_anchor_date_of_interest_payment=ActusDateTime(2024, 1, 1, 0, 0, 0),
    rate_reset_cycle="1W",  # Weekly resets
    rate_reset_anchor=ActusDateTime(2024, 1, 1, 0, 0, 0),
    delivery_settlement="D",  # Net settlement
    currency="USD",
)

rf_observer = ConstantRiskFactorObserver(0.03)
contract = create_contract(attrs, rf_observer)
```

**For a complete working example, see:**
- `examples/interest_rate_swap_example.py` - 5-year swap with overnight floating leg

### SWAPS - Generic Swap (Composition)

Generic swap contracts enable composition of multiple legs with different currencies, rates, and terms.

**Key Features:**
- References child contracts (legs)
- Supports multi-currency swaps
- Cross-currency basis swaps
- Net or gross settlement per leg
- Flexible leg composition

**Use Cases:**
- Cross-currency basis swaps
- Multi-leg interest rate swaps
- Complex swap structures
- Synthetic foreign currency borrowing

**Example:**
```python
from jactus.observers import MockChildContractObserver

# Create child legs (EUR floating and USD floating)
# ... (create EUR and USD leg contracts)

# Register child contracts
child_observer = MockChildContractObserver()
child_observer.register_child("EUR-LEG", events=eur_events, state=eur_state)
child_observer.register_child("USD-LEG", events=usd_events, state=usd_state)

# Create cross-currency swap
attrs = ContractAttributes(
    contract_id="XCCY-001",
    contract_type=ContractType.SWAPS,
    contract_role=ContractRole.RPA,
    status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
    maturity_date=ActusDateTime(2029, 1, 1, 0, 0, 0),
    contract_structure='{"FirstLeg": "EUR-LEG", "SecondLeg": "USD-LEG"}',
    delivery_settlement="D",
    currency="USD",
)

contract = create_contract(attrs, rf_observer, child_observer)
```

**For a complete working example, see:**
- `examples/cross_currency_basis_swap_example.py` - EUR/USD 5-year cross-currency basis swap

### CAPFL - Cap/Floor

Cap and floor contracts set upper (cap) or lower (floor) limits on floating interest rates.

**Key Features:**
- References underlier contract (typically swap)
- Cap strikes (upper limit)
- Floor strikes (lower limit)
- Protection against rate movements
- Periodic settlement

**Use Cases:**
- Interest rate protection
- Capped floating rate loans
- Collars (cap + floor combination)
- Hedging rate volatility

**Example:**
```python
# Interest rate cap at 6%
attrs = ContractAttributes(
    contract_id="CAP-001",
    contract_type=ContractType.CAPFL,
    contract_role=ContractRole.RPA,
    status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
    maturity_date=ActusDateTime(2029, 1, 1, 0, 0, 0),
    contract_structure='{"Underlying": "SWAP-001"}',
    rate_reset_cap=0.06,  # 6% cap
    currency="USD",
)

contract = create_contract(attrs, rf_observer, child_observer)
```

## Option Derivatives

### OPTNS - Options

Option contracts provide the right (but not obligation) to buy or sell an underlier at a specified strike price.

**Key Features:**
- Call or put options
- European or American exercise
- Strike prices (up to 2 for spreads)
- Option premium
- Settlement at exercise or maturity

**Use Cases:**
- Hedging directional risk
- Speculation on price movements
- Protective puts
- Covered calls
- Option spreads and strategies

**Example:**
```python
# European call option
attrs = ContractAttributes(
    contract_id="OPT-001",
    contract_type=ContractType.OPTNS,
    contract_role=ContractRole.RPA,
    status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
    maturity_date=ActusDateTime(2025, 1, 1, 0, 0, 0),
    option_type="C",  # Call
    option_strike_1=105_000.0,
    option_exercise_type="E",  # European
    settlement_currency="USD",
    currency="USD",
)

contract = create_contract(attrs, rf_observer)
```

### FUTUR - Futures

Futures contracts represent obligations to buy or sell an underlier at a specified future price and date.

**Key Features:**
- Future price agreed at inception
- Marked-to-market daily
- Settlement at maturity
- Standardized contracts

**Use Cases:**
- Price hedging
- Speculation on future prices
- Commodity hedging
- Currency hedging
- Interest rate futures

**Example:**
```python
# Futures contract
attrs = ContractAttributes(
    contract_id="FUT-001",
    contract_type=ContractType.FUTUR,
    contract_role=ContractRole.RPA,
    status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
    maturity_date=ActusDateTime(2024, 12, 31, 0, 0, 0),
    future_price=105_000.0,
    settlement_currency="USD",
    currency="USD",
)

contract = create_contract(attrs, rf_observer)
```

## Credit Derivatives

### CEG - Credit Enhancement Guarantee

Credit enhancement guarantees provide protection against credit events on covered contracts.

**Key Features:**
- References covered contracts
- Coverage ratio (percentage protected)
- Credit event detection
- Guarantee extent (notional, interest, or market value)
- Settlement on credit events

**Use Cases:**
- Credit risk mitigation
- Portfolio protection
- Loan guarantee programs
- Multi-name credit protection

**Example:**
```python
# Credit guarantee covering a loan at 80%
attrs = ContractAttributes(
    contract_id="CEG-001",
    contract_type=ContractType.CEG,
    contract_role=ContractRole.RPA,
    status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
    maturity_date=ActusDateTime(2029, 1, 1, 0, 0, 0),
    contract_structure='{"CoveredContract": "LOAN-001"}',
    coverage=0.8,  # 80% coverage
    credit_event_type="DL",  # Default
    credit_enhancement_guarantee_extent="NO",  # Notional only
    currency="USD",
)

contract = create_contract(attrs, rf_observer, child_observer)
```

### CEC - Credit Enhancement Collateral

Credit enhancement collateral contracts track collateral value against covered contract exposure.

**Key Features:**
- References covered and covering contracts
- Collateral sufficiency checking
- Coverage ratio requirement
- Collateral settlement (return or seizure)
- Multiple contracts supported

**Use Cases:**
- Margin accounts
- Collateralized lending
- Secured credit facilities
- Repo transactions

**Example:**
```python
# Collateral contract with 120% coverage requirement
attrs = ContractAttributes(
    contract_id="CEC-001",
    contract_type=ContractType.CEC,
    contract_role=ContractRole.RPA,
    status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
    maturity_date=ActusDateTime(2029, 1, 1, 0, 0, 0),
    contract_structure='{"CoveredContract": "LOAN-001", "CoveringContract": "STK-001"}',
    coverage=1.2,  # 120% collateral requirement
    credit_enhancement_guarantee_extent="NO",  # Notional only
    currency="USD",
)

contract = create_contract(attrs, rf_observer, child_observer)
```

## Contract Composition

Several derivative types support composition, where parent contracts reference child contracts:

### Composition Contract Types
- **SWAPS**: References leg contracts (FirstLeg, SecondLeg)
- **CAPFL**: References underlier contract (Underlying)
- **CEG**: References covered contracts (CoveredContract/CoveredContracts)
- **CEC**: References covered and covering contracts

### Child Contract Observer

Composition contracts require a `ChildContractObserver` to access child contract data:

```python
from jactus.observers import MockChildContractObserver

observer = MockChildContractObserver()

# Register child contract events, state, and attributes
observer.register_child(
    identifier="CHILD-001",
    events=[...],
    state=child_state,
    attributes={"notional_principal": 100000.0}
)

# Create parent contract with child observer
parent = create_contract(parent_attrs, rf_observer, observer)
```

## Settlement Options

Derivatives support different settlement modes:

- **`D` (Delivery/Net)**: Net settlement of cash flows
- **`S` (Physical/Gross)**: Gross settlement (both sides exchanged)

```python
# Net settlement (only difference paid)
delivery_settlement="D"

# Gross settlement (both currencies/amounts exchanged)
delivery_settlement="S"
```

## Running the Examples

The examples directory contains three comprehensive derivative examples:

### 1. Interest Rate Swap
```bash
python examples/interest_rate_swap_example.py
```

**Demonstrates:**
- 5-year fixed vs floating swap
- Overnight rate resets
- Quarterly payments
- Net settlement
- Scenario analysis

### 2. FX Swap
```bash
python examples/fx_swap_example.py
```

**Demonstrates:**
- EUR/USD currency pair
- 1-year forward
- Spot and forward rates
- Forward premium calculation
- Covered interest parity

### 3. Cross-Currency Basis Swap
```bash
python examples/cross_currency_basis_swap_example.py
```

**Demonstrates:**
- 5-year EUR vs USD swap
- Floating rates in both currencies
- Basis spread (30 bps)
- Contract composition (two legs)
- FX risk analysis

## Implementation Status

| Contract Type | Status | Tests | Coverage | Example |
|--------------|--------|-------|----------|---------|
| FXOUT | ✅ Complete | 20/20 | 100% | fx_swap_example.py |
| OPTNS | ✅ Complete | 25/25 | 97.20% | - |
| FUTUR | ✅ Complete | 20/20 | 97.22% | - |
| SWPPV | ✅ Complete | 22/22 | 95.12% | interest_rate_swap_example.py |
| SWAPS | ✅ Complete | 22/22 | 93.38% | cross_currency_basis_swap_example.py |
| CAPFL | ✅ Complete | 19/19 | 96.08% | - |
| CEG | ✅ Complete | 30/30 | 94.74% | - |
| CEC | ✅ Complete | 31/31 | 96.00% | - |

**Total:** 189 unit tests, all passing, 95%+ coverage

## Key Concepts

### JAX Compatibility
All derivatives are JAX-compatible:
- State variables as `jnp.ndarray`
- Automatic differentiation support
- Vectorization via `vmap`
- GPU acceleration ready

### Event Schedules
Derivatives generate event schedules including:
- **IED**: Initial Exchange Date
- **IP**: Interest Payment
- **RR**: Rate Reset
- **XD**: Exercise Date
- **STD**: Settlement
- **MD**: Maturity Date

### State Variables
Common state variables:
- `nt`: Notional Principal
- `ipnr`: Interest Payment Nominal Rate
- `ipac`: Interest Payment Accrued
- `sd`: Status Date
- `tmd`: Maturity Date
- `prf`: Performance (PF/DL/DQ/DF)

### Risk Factor Observers
All contracts require a `RiskFactorObserver`:
- Provide market data (rates, FX rates, prices)
- Can be constant or time-varying
- Used for rate resets and valuations

## Testing

Run derivative contract tests:

```bash
# All derivative tests
pytest tests/unit/contracts/test_fxout.py tests/unit/contracts/test_optns.py \
       tests/unit/contracts/test_futur.py tests/unit/contracts/test_swppv.py \
       tests/unit/contracts/test_swaps.py tests/unit/contracts/test_capfl.py \
       tests/unit/contracts/test_ceg.py tests/unit/contracts/test_cec.py -v

# Integration tests
pytest tests/integration/test_derivative_validation.py \
       tests/integration/test_derivative_composition.py -v

# With coverage
pytest tests/unit/contracts/ --cov=src/jactus/contracts --cov-report=term
```

## References

- ACTUS Technical Specification v1.1
  - Section 7.14: Foreign Exchange Outright (FXOUT)
  - Section 7.15: Options (OPTNS)
  - Section 7.16: Futures (FUTUR)
  - Section 7.11: Plain Vanilla Swap (SWPPV)
  - Section 7.12: Swap (SWAPS)
  - Section 7.13: Cap/Floor (CAPFL)
  - Section 7.17: Credit Enhancement Guarantee (CEG)
  - Section 7.18: Credit Enhancement Collateral (CEC)

## Further Reading

- **Interest Rate Swaps**: Hull, J. "Options, Futures, and Other Derivatives"
- **FX Markets**: King, M. "The Foreign Exchange Market"
- **Cross-Currency Swaps**: Flavell, R. "Swaps and Other Derivatives"
- **Credit Derivatives**: O'Kane, D. "Modelling Single-name and Multi-name Credit Derivatives"
- **ACTUS Standard**: https://www.actusfrf.org
