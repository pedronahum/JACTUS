# JACTUS - JAX ACTUS Financial Contracts

JACTUS is a JAX-based implementation of the ACTUS (Algorithmic Contract Types Unified Standards) specification. It provides high-performance simulation of 18 financial contract types with automatic differentiation support.

## Project Structure

- `src/jactus/` — Main library (contracts, core types, engine, observers, utilities)
- `tools/mcp-server/` — MCP server for AI assistant integration
- `tests/` — Test suite (unit, integration, cross-validation, property, performance)
- `examples/` — Python scripts and Jupyter notebooks
- `docs/` — Architecture guides and contract documentation

## MCP Server — USE TOOLS FIRST

When the JACTUS MCP server is available, **always use the `jactus_*` MCP tools** to discover contract types, get schemas, validate attributes, and simulate contracts. Do NOT read JACTUS source code to learn about contracts or parameters — the MCP tools are authoritative and return everything you need, including complete working Python examples.

Required workflow:
1. `jactus_list_contracts` — discover available contract types
2. `jactus_get_contract_schema` — get required/optional fields and example code
3. `jactus_validate_attributes` — verify attributes before simulation
4. `jactus_simulate_contract` — run the simulation
5. `jactus_list_risk_factor_observers` — discover market and behavioral observer types
6. `jactus_get_topic_guide` — structured guides on: `"contracts"`, `"behavioral"`, `"scenario"`, `"jax"`, `"events"`, `"attributes"`
7. `jactus_search_docs` — keyword search across documentation

For behavioral observers (prepayment, deposit transactions, callout events, scenarios), use `jactus_get_topic_guide("behavioral")` and `jactus_get_topic_guide("scenario")`.

Only read source code if you need to modify or debug JACTUS itself. For using JACTUS (simulating contracts, generating code, understanding parameters), the MCP tools are sufficient.

Run: `python -m jactus_mcp` (stdio) or `python -m jactus_mcp --transport streamable-http`

## Key Concepts

- **Contract Types**: PAM, LAM, LAX, NAM, ANN, CLM, UMP, CSH, STK, COM, FXOUT, OPTNS, FUTUR, SWPPV, SWAPS, CAPFL, CEG, CEC (18 total)
- **Risk Factor Observers**: `ConstantRiskFactorObserver`, `DictRiskFactorObserver`, `TimeSeriesRiskFactorObserver`, `CurveRiskFactorObserver`, `CompositeRiskFactorObserver`, `CallbackRiskFactorObserver`, `JaxRiskFactorObserver`
- **Behavioral Risk Factor Observers**: `PrepaymentSurfaceObserver`, `DepositTransactionObserver` — extend the observer framework with behavioral models that inject `CalloutEvent`s into the simulation timeline (protocol: `BehaviorRiskFactorObserver`, base class: `BaseBehaviorRiskFactorObserver`)
- **Scenario Management**: `Scenario` — defines simulation scenarios combining contract attributes with risk factor and behavioral observer configurations
- **Utilities**: `Surface2D`, `LabeledSurface2D` — 2D surface interpolation utilities used by behavioral observers (e.g., prepayment surfaces indexed by rate incentive and seasoning)
- **Events**: IED, IP, PR, MD, RR, etc. — represent cash flows and state transitions
- **ContractAttributes**: Pydantic model defining all contract parameters (`src/jactus/core/attributes.py`)

## Quick Start

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

rf_observer = ConstantRiskFactorObserver(constant_value=0.0)
contract = create_contract(attrs, rf_observer)
result = contract.simulate()
```

## Running Tests

```bash
# Main library tests
pytest tests/ -v

# MCP server tests
cd tools/mcp-server && pytest tests/ -v
```

## Conventions

- All financial amounts use Python `float` (backed by JAX `jnp.ndarray` internally)
- Dates use `ActusDateTime` (wraps Python datetime with ACTUS-specific methods)
- Cycles use string notation: `"1M"`, `"3M"`, `"6M"`, `"1Y"`
- Contract roles: `RPA` = lender/asset side, `RPL` = borrower/liability side
