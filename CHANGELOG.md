# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- MCP (Model Context Protocol) server for AI-assisted development
  - 10 tools for contract discovery, validation, examples, documentation, and system diagnostics
  - 4 MCP resources providing direct access to JACTUS documentation
  - 4 MCP prompts for common tasks (create contract, troubleshoot, understand, compare)
  - Full integration with Claude Desktop and Claude Code
  - 34 tests with 100% passing rate
  - Health check and version info tools for troubleshooting
  - Direct execution support via `python -m jactus_mcp`

## [0.1.0] - 2025-01-21

### Project Information
- **License**: Apache License 2.0
- **Author**: Pedro N. Rodriguez (pnrodriguezh@gmail.com)
- **Repository**: https://github.com/pedronahum/jactus

### Implemented Features

#### Core Infrastructure
- Type-safe implementation using Pydantic and mypy
- JAX-based computation for automatic differentiation and GPU acceleration
- Comprehensive exception hierarchy for error handling
- Flexible logging system with multiple output levels
- Complete test suite with property-based testing (Hypothesis)

#### Contract Types (18 Total - Complete ACTUS v1.1 Implementation)

**Principal Contracts (6)**
- **PAM** - Principal at Maturity (interest-only loans, bonds)
- **LAM** - Linear Amortizer (fixed principal amortization)
- **LAX** - Exotic Linear Amortizer (variable amortization schedules)
- **NAM** - Negative Amortizer (increasing principal balance)
- **ANN** - Annuity (mortgages, equal payment loans)
- **CLM** - Call Money (variable principal, on-demand repayment)

**Non-Principal Contracts (3)**
- **UMP** - Undefined Maturity Profile (revolving credit lines)
- **CSH** - Cash (money market accounts, escrow)
- **STK** - Stock (equity positions)

**Exotic Non-Principal Contracts (1)**
- **COM** - Commodity (physical commodities, futures underliers)

**Derivative Contracts (8)**
- **FXOUT** - Foreign Exchange Outright (FX forwards, swaps)
- **OPTNS** - Options (calls, puts, European/American)
- **FUTUR** - Futures (standardized forward contracts)
- **SWPPV** - Plain Vanilla Swap (fixed vs floating interest rate swaps)
- **SWAPS** - Generic Swap (cross-currency swaps, multi-leg swaps)
- **CAPFL** - Cap/Floor (interest rate caps and floors)
- **CEG** - Credit Enhancement Guarantee (credit protection)
- **CEC** - Credit Enhancement Collateral (collateral management)

#### Core Modules

**Time & Scheduling** (`jactus.core.time`, `jactus.utilities.schedules`)
- ActusDateTime for ACTUS-compliant date handling
- Schedule generation with cycle parsing (1D, 1W, 1M, 3M, 1Y, etc.)
- Business day conventions (SCF, SCMF, CSF, CSMF, SCP, SCMP, CSP, CSMP)
- End-of-month conventions (SD, EOM)
- Shift conventions for payment adjustments

**Contract Attributes** (`jactus.core.attributes`)
- 100+ ACTUS contract attributes
- Full validation using Pydantic
- Automatic mapping between ACTUS codes and Python names
- Support for all contract types and features

**State & Events** (`jactus.core.state`, `jactus.core.events`)
- ContractState with 20+ state variables (JAX arrays)
- EventSchedule with 30+ event types
- State transition tracking with ContractPerformance

**Contract Functions** (`jactus.functions`)
- Payoff functions (POF) for all contract types
- State transition functions (STF) for state evolution
- Shared utility functions for interest, principal, fees

**Risk Factor Observers** (`jactus.observers`)
- ConstantRiskFactorObserver for fixed rates
- MockChildContractObserver for composite contracts
- Protocol-based design for extensibility

**Simulation Engine** (`jactus.engine`)
- Event-driven simulation framework
- State progression through event timeline
- Support for composite contracts with child observers

#### Examples & Documentation

**Python Examples**
- `pam_example.py` - Comprehensive PAM demonstration (mortgages, bonds)
- `lam_example.py` - Linear amortization examples (auto loans, equipment financing)
- `interest_rate_swap_example.py` - 5-year fixed vs O/N floating swap
- `fx_swap_example.py` - EUR/USD FX swap with forward pricing
- `cross_currency_basis_swap_example.py` - 5-year EUR/USD basis swap
- `basic_example.py` - Simple getting started example

**Jupyter Notebooks**
- `01_annuity_mortgage.ipynb` - 30-year mortgage with amortization charts
- `02_options_contracts.ipynb` - Call/put options with payoff diagrams
- `03_interest_rate_cap.ipynb` - Interest rate protection scenarios
- `04_stock_commodity.ipynb` - Asset position tracking

**Documentation**
- Architecture guide (docs/ARCHITECTURE.md) - Complete system design
- PAM implementation walkthrough (docs/PAM.md) - Deep dive into internals
- Derivative contracts guide (docs/derivatives.md) - All 8 derivative types
- Sphinx documentation setup
- API reference documentation

**AI-Assisted Development**
- MCP server in `tools/mcp-server/` for Claude Desktop and Claude Code
- 8 tools for contract discovery, validation, examples, and documentation
- Full test suite with 24 tests (100% passing)

#### Testing & Quality

**Test Coverage**
- **Total Tests**: 1,192 tests (100% passing)
- **Unit Tests**: Comprehensive coverage of all modules
- **Integration Tests**: Contract composition and workflow tests
- **Property Tests**: Hypothesis-based testing for core functions
- **Performance Tests**: Benchmarking framework
- **Overall Coverage**: 95%+ across all contracts

**Quality Tools**
- Black for code formatting
- Ruff for linting (configured for finance domain)
- Mypy for static type checking
- Pre-commit hooks for automated checks
- pytest with coverage reporting

#### Development Workflow
- Makefile with common commands (test, lint, format, docs)
- Development scripts for setup and quality checks
- Comprehensive CONTRIBUTING.md guidelines
- GitHub issue templates and PR guidelines

### Technical Highlights

- **JAX Integration**: All numeric operations use JAX for GPU acceleration and automatic differentiation
- **Type Safety**: Full type annotations with mypy strict mode
- **ACTUS Compliance**: Implements ACTUS Technical Specification v1.1
- **Composability**: Support for composite contracts (SWAPS, CEG, CEC, CAPFL)
- **Performance**: JIT-compiled functions for production-grade performance
- **Extensibility**: Protocol-based design for easy extension

### Requirements

- Python >= 3.10, < 3.13
- JAX >= 0.4.20
- Flax >= 0.8.0
- NumPy >= 1.24.0
- Pydantic >= 2.5.0
- Python-dateutil >= 2.8.2

### Notes

- Complete implementation of ACTUS v1.1 specification
- All 18 contract types fully implemented and tested
- Production-ready with comprehensive documentation and examples

---

[Unreleased]: https://github.com/pedronahum/jactus/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/pedronahum/jactus/releases/tag/v0.1.0
