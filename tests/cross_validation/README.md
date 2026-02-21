# ACTUS Cross-Validation Tests

This directory contains cross-validation tests that verify JACTUS simulation output against the **official ACTUS test vectors** published by the ACTUS Financial Research Foundation.

## Test Source

All test vectors are sourced from the official ACTUS test repository:

> **https://github.com/actusfrf/actus-tests**

Each JSON file in [`data/`](data/) corresponds to one contract type and contains test cases with:
- **Contract terms** — input parameters (notional, rates, cycles, dates, etc.)
- **Market data** — observed risk factor time series (for rate resets, FX, etc.)
- **Expected results** — the reference event sequence with payoffs, notional, accrued interest, and nominal rate at each event

The files are downloaded automatically on first run from the `master` branch of the upstream repository and cached locally in `data/`.

## Results

JACTUS passes **276 out of 276** official test cases — **100% compliance** across all 18 contract types.

```
PAM  Cross-Validation:  25/25  passed    ANN  Cross-Validation:  31/31  passed
LAM  Cross-Validation:  31/31  passed    CLM  Cross-Validation:  15/15  passed
NAM  Cross-Validation:  22/22  passed    LAX  Cross-Validation:  18/18  passed
UMP  Cross-Validation:   9/9   passed    CSH  Cross-Validation:   4/4   passed
STK  Cross-Validation:  10/10  passed    COM  Cross-Validation:   4/4   passed
FXOUT Cross-Validation: 12/12  passed    OPTNS Cross-Validation: 23/23  passed
FUTUR Cross-Validation: 14/14  passed    SWPPV Cross-Validation: 14/14  passed
SWAPS Cross-Validation: 11/11  passed    CAPFL Cross-Validation:  4/4   passed
CEG  Cross-Validation:  14/14  passed    CEC  Cross-Validation:  15/15  passed
─────────────────────────────────────────────────────────────────────────────────
TOTAL                  276/276  passed    0 failures
```

No contract types are partially passing — every individual test case within every type is fully matched against the reference implementation on all four compared fields (payoff, notional principal, nominal interest rate, accrued interest).

## Coverage

All 18 ACTUS contract types implemented in JACTUS are covered:

| Category | Type | Description | Cases | Status |
|----------|------|-------------|------:|--------|
| Basic | PAM | Principal at Maturity | 25 | 25/25 |
| Basic | LAM | Linear Amortizer | 31 | 31/31 |
| Basic | NAM | Negative Amortizer | 22 | 22/22 |
| Basic | ANN | Annuity | 31 | 31/31 |
| Basic | CLM | Call Money | 15 | 15/15 |
| Basic | LAX | Exotic Linear Amortizer (array schedules) | 18 | 18/18 |
| Basic | UMP | Undefined Maturity Profile | 9 | 9/9 |
| Simple | CSH | Cash | 4 | 4/4 |
| Simple | STK | Stock | 10 | 10/10 |
| Simple | COM | Commodity | 4 | 4/4 |
| Derivatives | FXOUT | Foreign Exchange Outright | 12 | 12/12 |
| Derivatives | OPTNS | Options | 23 | 23/23 |
| Derivatives | FUTUR | Futures | 14 | 14/14 |
| Swaps | SWPPV | Plain Vanilla Interest Rate Swap | 14 | 14/14 |
| Swaps | SWAPS | Multi-leg Swap | 11 | 11/11 |
| Swaps | CAPFL | Cap/Floor | 4 | 4/4 |
| Credit | CEG | Credit Enhancement Guarantee | 14 | 14/14 |
| Credit | CEC | Credit Enhancement Collateral | 15 | 15/15 |
| | | **Total** | **276** | **276/276** |

## How It Works

### Architecture

```
conftest.py          — Downloads and caches test JSON files from GitHub
actus_mapper.py      — Maps ACTUS JSON field names to JACTUS ContractAttributes
                       (camelCase → snake_case, enum normalization, date parsing)
runner.py            — Core comparison engine: simulates contracts and compares
                       events against expected results
test_actus_*.py      — Pytest test classes grouped by contract category
```

### Comparison Logic

The test runner (`runner.py`) performs these steps for each test case:

1. **Parse** ACTUS JSON terms into a `ContractAttributes` instance
2. **Build** a `TimeSeriesRiskFactorObserver` from market data (if present)
3. **Build** a `SimulatedChildContractObserver` for composite contracts (SWAPS, CAPFL, CEG, CEC) by simulating child/leg contracts first
4. **Simulate** the contract using `create_contract()` + `contract.simulate()`
5. **Align** simulated events with expected events by `(date, event_type)` pairs
6. **Compare** numeric fields using combined tolerance:
   - Absolute tolerance: $1.00
   - Relative tolerance: 0.01%

Compared fields per event: `payoff`, `notionalPrincipal`, `nominalInterestRate`, `accruedInterest`.

## Running the Tests

From the project root:

```bash
# Run all cross-validation tests
pytest tests/cross_validation/ -v

# Run a specific contract type
pytest tests/cross_validation/test_actus_pam.py -v

# Run with stdout to see per-case pass/fail summaries
pytest tests/cross_validation/ -v -s

# Run alongside unit tests (full suite)
pytest tests/ -v
```

### First Run

On the first run, test JSON files are automatically downloaded from GitHub and cached in `data/`. Subsequent runs use the cached files. To force re-download, delete the files in `data/` and run again.

## File Organization

Test modules are grouped by contract category:

| Module | Contract Types |
|--------|---------------|
| `test_actus_pam.py` | PAM |
| `test_actus_principal.py` | LAM, NAM, ANN |
| `test_actus_clm.py` | CLM |
| `test_actus_exotic.py` | LAX, UMP |
| `test_actus_simple.py` | CSH, STK, COM |
| `test_actus_derivatives.py` | FXOUT, OPTNS, FUTUR |
| `test_actus_swaps.py` | SWPPV, SWAPS, CAPFL |
| `test_actus_credit.py` | CEG, CEC |

## Special Handling

Some contract types require additional processing beyond straightforward simulation:

- **Open-ended contracts** (UMP, STK, CLM) — maturity date is derived from the test case `to` field since these contracts have no natural maturity
- **CLM call events** — XD (exercise) events are injected from `eventsObserved`, and IP/MD events are shifted to the settlement date (XD + xDayNotice)
- **Composite contracts** (SWAPS, CAPFL, CEG, CEC) — child/leg contracts are parsed from `contractStructure`, simulated independently, and their results are fed into a `SimulatedChildContractObserver`
- **SWAPS** — FIL and SEL legs are simulated as separate ANN/PAM contracts; events are merged for net settlement (`DS=S`) or kept separate for gross settlement (`DS=D`)
- **CEG/CEC credit events** — credit events from `eventsObserved` are injected into child contract histories so the guarantee/collateral engine can detect defaults
