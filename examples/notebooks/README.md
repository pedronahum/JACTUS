# JACTUS Jupyter Notebook Examples

This directory contains interactive Jupyter notebooks demonstrating various ACTUS contract types implemented in JACTUS.

## Notebooks

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

- [JACTUS Documentation](https://docs.jactus.dev)
- [ACTUS Standard](https://www.actusfrf.org/)
- [GitHub Repository](https://github.com/pedronahum/jactus)

## License

Apache License 2.0
