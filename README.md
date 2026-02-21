# JACTUS

> High-performance implementation of the ACTUS financial contract standard using JAX

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

<p align="center">
  <img src="docs/jactus-demo.gif" alt="JACTUS Demo" width="800">
</p>

## Overview

JACTUS is a Python library that implements the **ACTUS (Algorithmic Contract Types Unified Standards)** specification using JAX for high-performance, differentiable financial contract modeling.

### Key Features

- **High Performance**: Leverages JAX's JIT compilation and GPU acceleration
- **Automatic Differentiation**: Built-in support for gradient-based risk analytics
- **Type Safety**: Full type annotations with mypy support
- **Comprehensive**: Implements the complete ACTUS standard
- **Well Tested**: 109/109 official ACTUS cross-validation test cases passing
- **Production Ready**: Robust error handling, logging, and documentation

## What is ACTUS?

ACTUS (Algorithmic Contract Types Unified Standards) is a standardized framework for representing financial contracts as mathematical algorithms. It provides a unified approach to modeling cash flows, risk analytics, and contract behavior across various financial instruments.

## Installation

### From GitHub

```bash
# Install directly from GitHub
pip install git+https://github.com/pedronahum/JACTUS.git

# Or clone and install locally
git clone https://github.com/pedronahum/JACTUS.git
cd JACTUS
pip install .
```

### Development Installation

```bash
git clone https://github.com/pedronahum/JACTUS.git
cd JACTUS
pip install -e ".[dev,docs,viz]"
```

### Requirements

- Python 3.10 or higher
- JAX >= 0.4.20
- Flax >= 0.8.0
- NumPy >= 1.24.0

## Quick Start

```python
from jactus.contracts import create_contract
from jactus.core import ContractAttributes, ContractType, ContractRole, ActusDateTime
from jactus.observers import ConstantRiskFactorObserver

# Create a simple Principal at Maturity (PAM) loan
# $100,000 loan at 5% interest, 1 year maturity
attrs = ContractAttributes(
    contract_id="LOAN-001",
    contract_type=ContractType.PAM,
    contract_role=ContractRole.RPA,  # We are the lender
    status_date=ActusDateTime(2024, 1, 1),
    initial_exchange_date=ActusDateTime(2024, 1, 15),
    maturity_date=ActusDateTime(2025, 1, 15),
    notional_principal=100_000.0,
    nominal_interest_rate=0.05,  # 5% annual
    cycle_of_interest_payment="6M",  # Semi-annual interest
    day_count_convention="30E360",
)

# Create risk factor observer (constant 5% rate)
rf_observer = ConstantRiskFactorObserver()

# Create and simulate the contract
contract = create_contract(attrs, rf_observer)
result = contract.simulate()

# Display cash flows
for event in result.events:
    if event.payoff != 0:
        print(f"{event.event_time}: {event.event_type.name:4s} ${event.payoff:>10,.2f}")

# Output:
# 2024-01-15: IED  $-100,000.00  (loan disbursement)
# 2024-07-15: IP   $  2,500.00   (6-month interest)
# 2025-01-15: MD   $102,500.00   (principal + final interest)
```

For more examples, see the [examples/](examples/) directory and [Jupyter notebooks](examples/notebooks/).

## Implemented Contract Types

JACTUS implements **18 ACTUS contract types** covering the complete ACTUS specification v1.1:

### Principal Contracts (6)
- **PAM** - Principal at Maturity (interest-only loans, bonds)
- **LAM** - Linear Amortizer (fixed principal amortization)
- **LAX** - Exotic Linear Amortizer (variable amortization schedules)
- **NAM** - Negative Amortizer (increasing principal balance)
- **ANN** - Annuity (mortgages, equal payment loans)
- **CLM** - Call Money (variable principal, on-demand repayment)

### Non-Principal Contracts (3)
- **UMP** - Undefined Maturity Profile (revolving credit lines)
- **CSH** - Cash (money market accounts, escrow)
- **STK** - Stock (equity positions)

### Exotic Non-Principal Contracts (1)
- **COM** - Commodity (physical commodities, futures underliers)

### Derivative Contracts (8)
- **FXOUT** - Foreign Exchange Outright (FX forwards, swaps)
- **OPTNS** - Options (calls, puts, European/American)
- **FUTUR** - Futures (standardized forward contracts)
- **SWPPV** - Plain Vanilla Swap (fixed vs floating interest rate swaps)
- **SWAPS** - Generic Swap (cross-currency swaps, multi-leg swaps)
- **CAPFL** - Cap/Floor (interest rate caps and floors)
- **CEG** - Credit Enhancement Guarantee (credit protection)
- **CEC** - Credit Enhancement Collateral (collateral management)

**Test Coverage:** 1,200+ unit/integration tests plus 109/109 official ACTUS cross-validation cases passing

## Documentation

### Core Documentation

- **[Architecture Guide](docs/ARCHITECTURE.md)** - Comprehensive system architecture, design patterns, and implementation details
- **[PAM Contract Walkthrough](docs/PAM.md)** - Deep dive into JACTUS internals using the Principal at Maturity contract
- **[Derivative Contracts Guide](docs/derivatives.md)** - Complete guide to all 8 derivative contract types

### API Documentation

Full API documentation is available at: [jactus.readthedocs.io](https://jactus.readthedocs.io) (coming soon)

### Building Documentation Locally

```bash
cd docs
make html
# Open docs/_build/html/index.html in your browser
```

## AI-Assisted Development

For Claude Code and LLM-assisted development, JACTUS includes an **MCP (Model Context Protocol) server** that provides AI assistants with direct access to JACTUS capabilities:

- **Contract Discovery**: List all 18 contract types with schemas
- **Example Retrieval**: Access working, tested code examples
- **Validation**: Validate contract attributes before creation
- **Documentation Search**: Search across all JACTUS docs
- **No Hallucination**: AI gets accurate information from your codebase

### Quick Setup

```bash
# Install MCP server
cd tools/mcp-server
pip install .
```

Configure in Claude Desktop (`~/Library/Application Support/Claude/claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "jactus": {
      "command": "python",
      "args": ["-m", "jactus_mcp.server"],
      "cwd": "/path/to/JACTUS"
    }
  }
}
```

See **[MCP Server Documentation](tools/mcp-server/README.md)** for full setup and usage.

## Development

### Setting Up Development Environment

```bash
# Run the setup script
./scripts/setup_dev.sh

# Or manually:
python3 -m venv venv
source venv/bin/activate
pip install -e ".[dev,docs,viz]"
pre-commit install
```

### Running Tests

```bash
# Run all tests
make test

# Run with coverage
make test-cov

# Run specific test markers
pytest -m unit
pytest -m integration
```

### Code Quality

```bash
# Format code
make format

# Run linter
make lint

# Type checking
make typecheck

# Run all quality checks
make quality
```

## Examples

### Interactive Jupyter Notebooks

Hands-on tutorials with visualizations in `examples/notebooks/`:

- **[01 - Annuity Mortgage](examples/notebooks/01_annuity_mortgage.ipynb)** - 30-year mortgage with amortization charts
- **[02 - Options Contracts](examples/notebooks/02_options_contracts.ipynb)** - Call/Put options with payoff diagrams
- **[03 - Interest Rate Cap](examples/notebooks/03_interest_rate_cap.ipynb)** - Interest rate protection scenarios
- **[04 - Stock & Commodity](examples/notebooks/04_stock_commodity.ipynb)** - Asset position tracking

### Python Scripts

Ready-to-run examples in `examples/`:

- `pam_example.py` - Principal at Maturity (bullet loans)
- `lam_example.py` - Linear Amortizer (equal principal payments)
- `interest_rate_swap_example.py` - Plain vanilla interest rate swap
- `fx_swap_example.py` - Foreign exchange swap
- `cross_currency_basis_swap_example.py` - Cross-currency basis swap

### Example Details

#### Interactive Notebooks (Recommended for Learning)

The Jupyter notebooks provide visual, hands-on learning with charts and step-by-step explanations:

- **Annuity (ANN)** - Mortgage amortization with payment composition visualization
- **Options (OPTNS)** - Call/put options with payoff diagrams
- **Cap/Floor (CAPFL)** - Interest rate protection analysis
- **Stock/Commodity (STK/COM)** - Position tracking and derivative underliers

#### Principal Contracts (Python Scripts)

- **[PAM Example](examples/pam_example.py)**: Comprehensive PAM (Principal at Maturity) examples
  - Basic loan simulation
  - Payment frequency comparison
  - Borrower vs. lender perspectives
  - JAX integration and sensitivity analysis
  - 30-year mortgage simulation
  - Portfolio analysis

- **[LAM Example](examples/lam_example.py)**: Comprehensive LAM (Linear Amortizer) examples
  - Basic amortizing loan with fixed principal payments
  - IPCB modes comparison (NT, NTIED, NTL)
  - Auto loan with monthly payments
  - LAM vs PAM comparison (interest savings)
  - Equipment financing with balloon payment
  - Portfolio of amortizing loans

#### Derivative Contracts

- **[Interest Rate Swap](examples/interest_rate_swap_example.py)**: Plain vanilla interest rate swap
  - 5-year fixed vs floating leg
  - Overnight (O/N) floating rate with weekly resets
  - Quarterly payment cycles
  - Net settlement demonstration
  - Market scenario analysis

- **[FX Swap](examples/fx_swap_example.py)**: EUR/USD foreign exchange swap
  - 1-year maturity
  - Spot and forward rate mechanics
  - Forward premium calculation
  - Covered interest parity demonstration
  - FX rate scenario analysis

- **[Cross-Currency Basis Swap](examples/cross_currency_basis_swap_example.py)**: EUR vs USD basis swap
  - 5-year tenor
  - 3M EURIBOR vs 3M SOFR + 30 bps basis
  - Multi-leg composition (SWAPS contract)
  - Dual currency floating rates
  - Basis spread impact analysis

Run examples:
```bash
# Principal contracts
python examples/pam_example.py  # Interest-only loans
python examples/lam_example.py  # Amortizing loans

# Derivative contracts
python examples/interest_rate_swap_example.py     # Interest rate swaps
python examples/fx_swap_example.py                 # FX swaps
python examples/cross_currency_basis_swap_example.py  # Cross-currency swaps
```

## Project Structure

```
jactus/
├── src/jactus/          # Main package source
│   ├── core/               # Core type definitions and enums
│   ├── utilities/          # Date/time and calendar utilities
│   ├── functions/          # Payoff and state transition functions
│   ├── observers/          # Risk factor observers
│   ├── engine/             # Event generation and simulation engines
│   ├── contracts/          # 18 ACTUS contract implementations
│   │   ├── base.py         # BaseContract abstract class
│   │   ├── pam.py          # Principal at Maturity
│   │   ├── lam.py          # Linear Amortizer
│   │   ├── lax.py          # Exotic Linear Amortizer
│   │   ├── nam.py          # Negative Amortizer
│   │   ├── ann.py          # Annuity
│   │   ├── clm.py          # Call Money
│   │   ├── ump.py          # Undefined Maturity Profile
│   │   ├── csh.py          # Cash
│   │   ├── stk.py          # Stock
│   │   ├── com.py          # Commodity
│   │   ├── fxout.py        # FX Outright
│   │   ├── optns.py        # Options
│   │   ├── futur.py        # Futures
│   │   ├── swppv.py        # Plain Vanilla Swap
│   │   ├── swaps.py        # Generic Swap
│   │   ├── capfl.py        # Cap/Floor
│   │   ├── ceg.py          # Credit Enhancement Guarantee
│   │   ├── cec.py          # Credit Enhancement Collateral
│   │   └── __init__.py     # Factory pattern and registry
│   ├── exceptions.py       # Custom exceptions
│   └── logging_config.py   # Logging configuration
├── tests/                  # Test suite (1,200+ tests, 95%+ coverage)
│   ├── unit/               # Unit tests for each module
│   ├── integration/        # Integration and end-to-end tests
│   ├── cross_validation/   # 109/109 official ACTUS test cases
│   ├── property/           # Property-based tests (Hypothesis)
│   └── performance/        # Performance benchmarks
├── docs/                   # Documentation
│   ├── ARCHITECTURE.md     # System architecture guide
│   ├── PAM.md              # PAM implementation walkthrough
│   └── derivatives.md      # Derivative contracts guide
├── examples/               # Example scripts
│   ├── pam_example.py                        # Principal at Maturity examples
│   ├── lam_example.py                        # Linear Amortizer examples
│   ├── interest_rate_swap_example.py         # Interest rate swap
│   ├── fx_swap_example.py                    # FX swap
│   ├── cross_currency_basis_swap_example.py  # Cross-currency swap
│   └── basic_example.py                      # Basic usage
└── scripts/                # Development scripts
```

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### Development Workflow

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Run tests and quality checks (`make all`)
5. Commit your changes (`git commit -m 'Add amazing feature'`)
6. Push to the branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## Citation

If you use JACTUS in your research, please cite:

```bibtex
@software{jactus,
  title = {JACTUS: High-performance ACTUS implementation using JAX},
  author = {Rodriguez, Pedro N.},
  year = {2025},
  url = {https://github.com/pedronahum/jactus}
}
```

### ACTUS Standard Citation

```bibtex
@techreport{actus2020,
  title = {ACTUS Technical Specification v1.1},
  author = {ACTUS Financial Research Foundation},
  year = {2020},
  url = {https://www.actusfrf.org/}
}
```

## Acknowledgments

- [ACTUS Financial Research Foundation](https://www.actusfrf.org/) for the ACTUS standard
- [Google JAX Team](https://github.com/google/jax) for the JAX framework
- All contributors to this project

## Project Status

**Release**: v0.1.0 - Complete implementation of ACTUS v1.1 specification ✅

- ✅ 18 contract types implemented
- ✅ 109/109 official ACTUS cross-validation test cases passing (PAM, LAM, NAM, ANN)
- ✅ 1,200+ unit/integration/property tests
- ✅ Full JAX integration with automatic differentiation
- ✅ Production-ready with comprehensive documentation
- ✅ Apache License 2.0

## Support

- **Issues**: [GitHub Issues](https://github.com/pedronahum/jactus/issues)
- **Discussions**: [GitHub Discussions](https://github.com/pedronahum/jactus/discussions)
- **Email**: pnrodriguezh@gmail.com

## Links

- [ACTUS Standard](https://www.actusfrf.org/)
- [JAX Documentation](https://jax.readthedocs.io/)
- [Flax Documentation](https://flax.readthedocs.io/)
- [Project Documentation](https://jactus.readthedocs.io)
