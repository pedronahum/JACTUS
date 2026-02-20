# JACTUS - JAX ACTUS Financial Contracts

JACTUS is a JAX-based implementation of the ACTUS (Algorithmic Contract Types Unified Standards) specification. It provides high-performance simulation of 18 financial contract types with automatic differentiation support.

## Project Structure

- `src/jactus/` — Main library (contracts, core types, engine, observers, utilities)
- `tools/mcp-server/` — MCP server for AI assistant integration
- `tests/` — Test suite (unit, integration, cross-validation, property, performance)
- `examples/` — Python scripts and Jupyter notebooks
- `docs/` — Architecture guides and contract documentation

## MCP Server

The MCP server (`tools/mcp-server/`) provides AI assistants with:
- **Contract discovery** — list types, get schemas, view event types
- **Contract simulation** — create and simulate contracts, get structured cash flows
- **Documentation search** — search across all JACTUS docs
- **Example retrieval** — get and run example code
- **Validation** — validate contract attributes before simulation

Run: `python -m jactus_mcp` (stdio) or `python -m jactus_mcp --transport streamable-http`

## Key Concepts

- **Contract Types**: PAM, LAM, LAX, NAM, ANN, CLM, UMP, CSH, STK, COM, FXOUT, OPTNS, FUTUR, SWPPV, SWAPS, CAPFL, CEG, CEC (18 total)
- **Risk Factor Observers**: `ConstantRiskFactorObserver`, `DictRiskFactorObserver`, `JaxRiskFactorObserver`
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
