# JACTUS — Project Context

JACTUS is a Python library implementing the ACTUS (Algorithmic Contract Types
Unified Standards) specification using JAX for high-performance, differentiable
financial contract modeling.

## Project Layout

- `src/jactus/` — main package
  - `core/` — ContractAttributes, ContractType, ActusDateTime enums
  - `contracts/` — 18 contract implementations (pam.py, ann.py, swppv.py, ...)
  - `observers/` — RiskFactorObserver base class + ConstantRiskFactorObserver
  - `engine/` — simulation engine
  - `functions/` — payoff and state transition functions
- `tools/mcp-server/` — MCP server exposing JACTUS via Model Context Protocol
- `skills/jactus/` — Agent Skill for Gemini CLI and compatible agents
- `tests/` — 1,400+ tests, 100% passing
- `examples/` — runnable Python scripts and Jupyter notebooks
- `docs/` — architecture guides and contract documentation

## Key Classes

```python
from jactus.contracts import create_contract
from jactus.core import ContractAttributes, ContractType, ContractRole, ActusDateTime
from jactus.observers import ConstantRiskFactorObserver
```

## Quick Simulation

```python
attrs = ContractAttributes(
    contract_id="X",
    contract_type=ContractType.PAM,
    contract_role=ContractRole.RPA,
    status_date=ActusDateTime(2024, 1, 1),
    initial_exchange_date=ActusDateTime(2024, 1, 15),
    maturity_date=ActusDateTime(2025, 1, 15),
    notional_principal=100_000.0,
    nominal_interest_rate=0.05,
)
result = create_contract(attrs, ConstantRiskFactorObserver()).simulate()
for event in result.events:
    if event.payoff != 0:
        print(f"{event.event_time} | {event.event_type.name} | {event.payoff:,.2f}")
```

## Supported Contract Types

**Principal:** PAM, LAM, LAX, NAM, ANN, CLM
**Non-Principal:** UMP, CSH, STK, COM
**Derivatives:** FXOUT, OPTNS, FUTUR, SWPPV, SWAPS, CAPFL, CEG, CEC

## Contract Roles

- `RPA` — Real Position Asset (lender/long)
- `RPL` — Real Position Liability (borrower/short)
- `RFL`/`PFL` — Receive/Pay First Leg (swaps)
- `BUY`/`SEL` — Protection Buyer/Seller (credit enhancement)
- `LG`/`ST` — Long/Short (futures)

## Risk Factor Observers

```python
# Fixed rate
rf = ConstantRiskFactorObserver(constant_value=0.0)

# Multiple static rates
rf = DictRiskFactorObserver({"LIBOR-3M": 0.05, "USD/EUR": 1.18})

# Time-varying rates
rf = TimeSeriesRiskFactorObserver(
    time_series={"LIBOR-3M": [(ActusDateTime(2024, 1, 1), 0.04), ...]},
    interpolation="step",
)
```

## JAX Risk Analytics

```python
import jax

def pv(rate):
    attrs = ContractAttributes(..., nominal_interest_rate=float(rate))
    return sum(e.payoff for e in create_contract(attrs, rf).simulate().events)

dv01 = jax.grad(pv)(0.05)  # automatic differentiation
```

## Portfolio Batch Simulation

```python
from jactus.contracts.portfolio import simulate_portfolio
results = simulate_portfolio([(attrs1, rf), (attrs2, rf), ...])
```

## Development Commands

- `pytest tests/ -v` — run full test suite
- `python -m jactus_mcp` — start MCP server
- `jactus contract list` — CLI: list contract types
- `jactus simulate --type PAM --notional 100000 --rate 0.05` — CLI: simulate

## Google Workspace Integration

JACTUS pairs with the `gws` CLI (https://github.com/googleworkspace/cli):
- Use `gws drive` to fetch term sheets
- Use JACTUS to simulate contract cash flows
- Use `gws sheets` to write results
- Use `gws gmail` to send summaries

## MCP Configuration

```json
{
  "mcpServers": {
    "jactus": { "command": "python", "args": ["-m", "jactus_mcp"] }
  }
}
```

## Key Conventions

- Dates: `ActusDateTime(YYYY, MM, DD)` with integer arguments
- Cycles: `"1M"` (monthly), `"3M"` (quarterly), `"6M"` (semi-annual), `"1Y"` (annual)
- Amounts: Python `float` (JAX `jnp.ndarray` internally)
- Day count: `AA`, `A360`, `A365`, `E30360ISDA`, `E30360`, `B30360`, `BUS252`
