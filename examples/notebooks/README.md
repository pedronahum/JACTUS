# JACTUS Jupyter Notebook Examples

This directory contains interactive Jupyter notebooks demonstrating various ACTUS contract types implemented in JACTUS.

## Notebooks

### 00 - Getting Started with PAM (`00_getting_started_pam.ipynb`)
**Contract Type:** PAM (Principal at Maturity)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pedronahum/JACTUS/blob/main/examples/notebooks/00_getting_started_pam.ipynb)

The quickest way to get started with JACTUS:
- Install JACTUS from PyPI
- Create a simple PAM loan ($100,000 at 5% for 1 year)
- Simulate and inspect all cash flow events
- Understand the ACTUS event lifecycle

**Key Learning:** How ACTUS models financial contracts as event-driven simulations with state transitions.

---

### 01 - Annuity Mortgage (`01_annuity_mortgage.ipynb`)
**Contract Type:** ANN (Annuity)

Demonstrates a 30-year fixed-rate mortgage with:
- Constant monthly payments
- Amortization schedule analysis
- Principal vs interest breakdown
- Visualization of payment composition

**Key Learning:** How annuity contracts handle fixed periodic payments with changing principal/interest ratios.

---

### 02 - Options Contracts (`02_options_contracts.ipynb`)
**Contract Type:** OPTNS (Options)

Covers European call and put options with:
- Call option (right to buy) examples
- Put option (right to sell) examples
- Payoff diagrams
- Option valuation at maturity

**Key Learning:** How options provide asymmetric payoff profiles and the difference between calls and puts.

---

### 03 - Interest Rate Cap (`03_interest_rate_cap.ipynb`)
**Contract Type:** CAPFL (Cap/Floor)

Demonstrates interest rate protection with:
- Interest rate cap on floating rate loan
- Protection value calculation
- Payoff scenarios
- Cost-benefit analysis

**Key Learning:** How caps/floors protect against adverse interest rate movements.

---

### 04 - Stock and Commodity (`04_stock_commodity.ipynb`)
**Contract Types:** STK (Stock), COM (Commodity)

Shows basic asset position tracking with:
- Stock position (AAPL example)
- Commodity position (Gold example)
- Price movement impact
- Use as underliers for derivatives

**Key Learning:** How STK and COM contracts serve as building blocks for derivative contracts.

---

## Running the Notebooks

### Prerequisites

```bash
pip install jactus
pip install jupyter matplotlib numpy
```

### Launch Jupyter

```bash
cd examples/notebooks
jupyter notebook
```

Then open any `.ipynb` file in your browser.

## Python Script Examples

For non-interactive examples, see the Python scripts in the parent `examples/` directory:

- `pam_example.py` - Principal at Maturity contract
- `lam_example.py` - Linear Amortizer contract
- `interest_rate_swap_example.py` - Plain vanilla swap
- `fx_swap_example.py` - FX swap
- `cross_currency_basis_swap_example.py` - Cross-currency swap

## ACTUS Contract Types

These notebooks demonstrate contract types from the **ACTUS Financial Research Foundation** standard v1.1:

| Code | Name | Description | Notebook |
|------|------|-------------|----------|
| ANN | Annuity | Fixed periodic payments | 01 |
| OPTNS | Options | Call/Put options | 02 |
| CAPFL | Cap/Floor | Interest rate protection | 03 |
| STK | Stock | Equity positions | 04 |
| COM | Commodity | Commodity positions | 04 |
| PAM | Principal at Maturity | Bullet loans/bonds | ../pam_example.py |
| LAM | Linear Amortizer | Equal principal payments | ../lam_example.py |
| SWPPV | Plain Vanilla Swap | Interest rate swap | ../interest_rate_swap_example.py |
| SWAPS | Generic Swap | Composite swap | Multiple examples |
| FXOUT | FX Outright | Foreign exchange | ../fx_swap_example.py |

## Learn More

- [JACTUS Documentation](https://pedronahum.github.io/JACTUS/)
- [ACTUS Standard](https://www.actusfrf.org/)
- [GitHub Repository](https://github.com/pedronahum/JACTUS)

## License

Apache License 2.0
