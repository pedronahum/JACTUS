# JACTUS

> High-performance implementation of the ACTUS financial contract standard using JAX

[![PyPI](https://img.shields.io/pypi/v/jactus)](https://pypi.org/project/jactus/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Documentation](https://img.shields.io/badge/docs-GitHub%20Pages-blue)](https://pedronahum.github.io/JACTUS/)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pedronahum/JACTUS/blob/main/examples/notebooks/00_getting_started_pam.ipynb)

<p align="center">
  <img src="docs/jactus-demo.gif" alt="JACTUS Demo">
</p>

## Overview

JACTUS is a Python library that implements the **ACTUS (Algorithmic Contract Types Unified Standards)** specification using JAX for high-performance, differentiable financial contract modeling.

### Key Features

- **High Performance**: Leverages JAX's JIT compilation and GPU acceleration
- **Automatic Differentiation**: Built-in support for gradient-based risk analytics
- **Behavioral Risk Models**: Prepayment surfaces, deposit behavior, and dynamic event injection via callout events
- **Scenario Management**: Bundle market and behavioral observers into named configurations
- **Type Safety**: Full type annotations with mypy support
- **Comprehensive**: Implements the complete ACTUS standard
- **Well Tested**: 276 official ACTUS cross-validation test cases passing across all 18 contract types
- **Production Ready**: Robust error handling, logging, and documentation

## What is ACTUS?

ACTUS (Algorithmic Contract Types Unified Standards) is a standardized framework for representing financial contracts as mathematical algorithms. It provides a unified approach to modeling cash flows, risk analytics, and contract behavior across various financial instruments.

## Installation

```bash
pip install jactus
```

**Requirements:** Python 3.10+, JAX >= 0.4.20

### GPU / TPU Acceleration

JACTUS runs on CPU by default. To enable hardware acceleration, install the
appropriate JAX backend **before** or **after** installing JACTUS:

```bash
# NVIDIA GPU (CUDA 13 — recommended)
pip install "jax[cuda13]"

# NVIDIA GPU (CUDA 12)
pip install "jax[cuda12]"

# Google Cloud TPU
pip install "jax[tpu]"
```

No code changes are required — JACTUS automatically detects the available
backend and selects the optimal execution strategy (e.g. `vmap` on GPU/TPU,
manual batching on CPU).

> **Precision note:** The array-mode simulation path uses float32 for
> performance. TPUs do not support float64. For CPU/GPU workloads requiring
> full double precision, enable it before importing JACTUS:
> `jax.config.update("jax_enable_x64", True)`

For development:

```bash
git clone https://github.com/pedronahum/JACTUS.git
cd JACTUS
pip install -e ".[dev,docs,viz]"
```

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
    interest_payment_cycle="6M",  # Semi-annual interest
    day_count_convention="30E360",
)

# Create risk factor observer
rf_observer = ConstantRiskFactorObserver(constant_value=0.0)

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

**Test Coverage:** 1,200+ unit/integration tests plus 276 official ACTUS cross-validation cases passing across all 18 contract types

## Risk Factor and Behavioral Observers

JACTUS provides a layered observer framework for market data and behavioral modeling:

### Market Risk Factor Observers
- **ConstantRiskFactorObserver** - Fixed constant value for all risk factors
- **DictRiskFactorObserver** - Per-identifier static values
- **TimeSeriesRiskFactorObserver** - Time-varying market data with step/linear interpolation
- **CurveRiskFactorObserver** - Yield/rate curves keyed by tenor
- **CompositeRiskFactorObserver** - Priority-based fallback across multiple observers
- **CallbackRiskFactorObserver** - Delegates to user-provided callables
- **JaxRiskFactorObserver** - Differentiable JAX-native observer for gradient-based analytics

### Behavioral Risk Factor Observers
- **BehaviorRiskFactorObserver** protocol and **BaseBehaviorRiskFactorObserver** ABC for custom behavioral models
- **PrepaymentSurfaceObserver** - 2D surface-based prepayment model (spread x loan age -> prepayment rate)
- **DepositTransactionObserver** - Deposit transaction behavior model for UMP contracts
- **CalloutEvent** - Dynamic event injection allowing behavioral observers to add events to the simulation timeline

### Scenario Management
- **Scenario** - Bundle market and behavioral observers into named configurations for reproducible analysis
- **Surface2D** / **LabeledSurface2D** - JAX-compatible 2D surface interpolation utilities

## Documentation

Full documentation is available at **[pedronahum.github.io/JACTUS](https://pedronahum.github.io/JACTUS/)**, including API reference, user guides, and the ACTUS specification overview.

### Core Documentation

- **[Architecture Guide](docs/ARCHITECTURE.md)** - Comprehensive system architecture, design patterns, and implementation details
- **[PAM Contract Walkthrough](docs/PAM.md)** - Deep dive into JACTUS internals using the Principal at Maturity contract
- **[Derivative Contracts Guide](docs/derivatives.md)** - Complete guide to all 8 derivative contract types

### Try It Now

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pedronahum/JACTUS/blob/main/examples/notebooks/00_getting_started_pam.ipynb)

Get started immediately with the **Getting Started** notebook — no local installation required.

### Building Documentation Locally

```bash
pip install -e ".[docs]"
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
# Install MCP server (automatically installs jactus from PyPI)
pip install git+https://github.com/pedronahum/JACTUS.git#subdirectory=tools/mcp-server
```

Configure in Claude Desktop (`~/Library/Application Support/Claude/claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "jactus": {
      "command": "python",
      "args": ["-m", "jactus_mcp"]
    }
  }
}
```

For Claude Code, the `.mcp.json` in the project root provides auto-discovery when you open the JACTUS workspace.

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

- **[00 - Getting Started (PAM)](examples/notebooks/00_getting_started_pam.ipynb)** - Quick start with a PAM contract [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pedronahum/JACTUS/blob/main/examples/notebooks/00_getting_started_pam.ipynb)
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
│   ├── observers/          # Risk factor and behavioral observers
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
│   ├── cross_validation/   # 276 official ACTUS cross-validation cases
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
  url = {https://github.com/pedronahum/JACTUS}
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

**Release**: v0.1.2 - Complete implementation of ACTUS v1.1 specification ✅

- ✅ 18 contract types implemented
- ✅ 276 official ACTUS cross-validation test cases passing across all 18 contract types
- ✅ 1,200+ unit/integration/property tests
- ✅ Full JAX integration with automatic differentiation
- ✅ Production-ready with comprehensive documentation
- ✅ Available on [PyPI](https://pypi.org/project/jactus/)
- ✅ Apache License 2.0

## Support

- **Issues**: [GitHub Issues](https://github.com/pedronahum/JACTUS/issues)
- **Discussions**: [GitHub Discussions](https://github.com/pedronahum/JACTUS/discussions)
- **Email**: pnrodriguezh@gmail.com

## Links

- [Project Documentation](https://pedronahum.github.io/JACTUS/)
- [PyPI Package](https://pypi.org/project/jactus/)
- [ACTUS Standard](https://www.actusfrf.org/)
- [JAX Documentation](https://jax.readthedocs.io/)
- [Flax Documentation](https://flax.readthedocs.io/)
