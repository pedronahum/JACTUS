# Google Workspace Integration Recipes

These recipes combine the `gws` CLI (Google Workspace CLI) with JACTUS for end-to-end
financial contract workflows. Each recipe reads data from Google Workspace, simulates
contracts with JACTUS, and writes results back to Workspace.

## Prerequisites

```bash
# Install and authenticate gws CLI
pip install gws
gws auth setup

# Install JACTUS
pip install git+https://github.com/pedronahum/JACTUS.git
```

---

## Recipe 1: Mortgage Analyzer

Read loan terms from a Google Doc, simulate an ANN (annuity) contract, and write
the full amortization schedule to Google Sheets.

### Steps

1. **Fetch loan terms from Google Docs**

```bash
gws docs get --document-id "YOUR_DOC_ID" --format text > loan_terms.txt
```

2. **Parse terms and simulate with JACTUS**

```python
import json
from jactus.contracts import create_contract
from jactus.core import ContractAttributes, ContractType, ContractRole, ActusDateTime
from jactus.observers import ConstantRiskFactorObserver

# Parse extracted terms (from agent or manual parsing)
principal = 400_000.0
rate = 0.065
term_years = 30

attrs = ContractAttributes(
    contract_id="MORTGAGE-001",
    contract_type=ContractType.ANN,
    contract_role=ContractRole.RPA,
    status_date=ActusDateTime(2024, 1, 1),
    initial_exchange_date=ActusDateTime(2024, 2, 1),
    maturity_date=ActusDateTime(2024 + term_years, 2, 1),
    notional_principal=principal,
    nominal_interest_rate=rate,
    principal_redemption_cycle="1M",
)

rf = ConstantRiskFactorObserver(constant_value=0.0)
contract = create_contract(attrs, rf)
result = contract.simulate()

# Build amortization table
rows = [["Date", "Event", "Payment", "Principal", "Interest", "Balance"]]
for event, state in zip(result.events, result.states):
    if event.payoff != 0:
        rows.append([
            str(event.event_time),
            event.event_type.name,
            f"{event.payoff:.2f}",
            f"{state.notional_principal:.2f}",
            f"{state.accrued_interest:.2f}",
            f"{state.notional_principal:.2f}",
        ])

# Save as CSV for Sheets upload
import csv
with open("amortization.csv", "w", newline="") as f:
    csv.writer(f).writerows(rows)
```

3. **Write amortization schedule to Google Sheets**

```bash
gws sheets create --title "Mortgage Amortization - MORTGAGE-001"
gws sheets import --spreadsheet-id "NEW_SHEET_ID" --file amortization.csv --sheet "Amortization"
```

### Expected Output

A Google Sheet with columns: Date, Event, Payment, Principal, Interest, Balance —
one row per payment event over the life of the mortgage.

---

## Recipe 2: Swap Book DV01 Report

Read a CSV of interest rate swaps from Google Drive, batch simulate SWPPV contracts,
compute DV01 for each, and email a summary via Gmail.

### Steps

1. **Download swap book from Drive**

```bash
gws drive download --file-id "YOUR_CSV_FILE_ID" --output swaps.csv
```

2. **Batch simulate and compute DV01**

```python
import csv
import json
from jactus.contracts import create_contract
from jactus.core import ContractAttributes, ContractType, ContractRole, ActusDateTime
from jactus.observers import ConstantRiskFactorObserver

def compute_dv01(attrs, base_rate=0.05, bump=0.0001):
    """Compute DV01 via finite difference."""
    rf_base = ConstantRiskFactorObserver(constant_value=base_rate)
    rf_bump = ConstantRiskFactorObserver(constant_value=base_rate + bump)

    pv_base = sum(e.payoff for e in create_contract(attrs, rf_base).simulate().events)
    pv_bump = sum(e.payoff for e in create_contract(attrs, rf_bump).simulate().events)

    return (pv_bump - pv_base) / bump

# Read swap book
results = []
with open("swaps.csv") as f:
    reader = csv.DictReader(f)
    for row in reader:
        attrs = ContractAttributes(
            contract_id=row["contract_id"],
            contract_type=ContractType.SWPPV,
            contract_role=ContractRole.RFL,
            status_date=ActusDateTime(2024, 1, 1),
            initial_exchange_date=ActusDateTime(
                int(row["start_year"]), int(row["start_month"]), 1
            ),
            maturity_date=ActusDateTime(
                int(row["end_year"]), int(row["end_month"]), 1
            ),
            notional_principal=float(row["notional"]),
            nominal_interest_rate=float(row["fixed_rate"]),
            nominal_interest_rate_2=float(row["floating_rate"]),
            interest_payment_cycle=row.get("payment_cycle", "6M"),
            rate_reset_cycle=row.get("reset_cycle", "3M"),
        )

        dv01 = compute_dv01(attrs)
        total_pv = sum(
            e.payoff
            for e in create_contract(
                attrs, ConstantRiskFactorObserver(0.0)
            ).simulate().events
        )

        results.append({
            "contract_id": row["contract_id"],
            "notional": row["notional"],
            "dv01": f"{dv01:.4f}",
            "net_pv": f"{total_pv:.2f}",
        })

# Write summary
summary = "Swap Book DV01 Report\n\n"
summary += f"{'ID':<15} {'Notional':>15} {'DV01':>10} {'Net PV':>15}\n"
summary += "-" * 60 + "\n"
for r in results:
    summary += f"{r['contract_id']:<15} {r['notional']:>15} {r['dv01']:>10} {r['net_pv']:>15}\n"

total_dv01 = sum(float(r["dv01"]) for r in results)
summary += f"\nPortfolio DV01: {total_dv01:.4f}\n"

with open("dv01_report.txt", "w") as f:
    f.write(summary)
```

3. **Email report via Gmail**

```bash
gws gmail send \
  --to "risk-team@example.com" \
  --subject "Daily Swap Book DV01 Report" \
  --body-file dv01_report.txt
```

### Expected Output

An email containing a table of swap contracts with their notional, DV01, and net PV,
plus the aggregate portfolio DV01.

---

## Recipe 3: FX Exposure Report

Pull FX forward terms from Google Sheets, simulate FXOUT contracts, and write
mark-to-market results to a new Sheet.

### Steps

1. **Read FX positions from Sheets**

```bash
gws sheets read --spreadsheet-id "YOUR_SHEET_ID" --range "FXPositions!A1:H50" --format csv > fx_positions.csv
```

2. **Simulate FX forwards**

```python
import csv
from jactus.contracts import create_contract
from jactus.core import ContractAttributes, ContractType, ContractRole, ActusDateTime
from jactus.observers import DictRiskFactorObserver

results = []
with open("fx_positions.csv") as f:
    reader = csv.DictReader(f)
    for row in reader:
        attrs = ContractAttributes(
            contract_id=row["deal_id"],
            contract_type=ContractType.FXOUT,
            contract_role=ContractRole.RPA,
            status_date=ActusDateTime(2024, 6, 1),
            initial_exchange_date=ActusDateTime(
                int(row["trade_year"]), int(row["trade_month"]), int(row["trade_day"])
            ),
            maturity_date=ActusDateTime(
                int(row["settle_year"]), int(row["settle_month"]), int(row["settle_day"])
            ),
            notional_principal=float(row["amount_ccy1"]),
            currency=row["ccy1"],
            currency_2=row["ccy2"],
            notional_principal_2=float(row["amount_ccy2"]),
            delivery_settlement="S",
        )

        # Current spot rate for mark-to-market
        current_spot = float(row["current_spot"])
        rf = DictRiskFactorObserver({f"{row['ccy1']}/{row['ccy2']}": current_spot})
        result = create_contract(attrs, rf).simulate()
        net_pv = sum(e.payoff for e in result.events)

        results.append([
            row["deal_id"], row["ccy1"], row["ccy2"],
            row["amount_ccy1"], row["amount_ccy2"],
            f"{current_spot:.4f}", f"{net_pv:.2f}",
        ])

# Write CSV for Sheets
with open("fx_mtm.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Deal ID", "CCY1", "CCY2", "Amount CCY1", "Amount CCY2", "Spot", "MTM"])
    writer.writerows(results)
```

3. **Write MTM results to a new Sheet**

```bash
gws sheets create --title "FX Exposure Report - $(date +%Y-%m-%d)"
gws sheets import --spreadsheet-id "NEW_SHEET_ID" --file fx_mtm.csv --sheet "MTM"
```

### Expected Output

A Google Sheet with columns: Deal ID, CCY1, CCY2, Amount CCY1, Amount CCY2,
Current Spot, Mark-to-Market value.

---

## Recipe 4: Options Dashboard

Read options positions from Google Drive, compute delta (first-order sensitivity)
via JAX automatic differentiation, and write greeks to Sheets.

### Steps

1. **Download options book from Drive**

```bash
gws drive download --file-id "YOUR_OPTIONS_CSV_ID" --output options.csv
```

2. **Compute delta for each option**

```python
import csv
from jactus.contracts import create_contract
from jactus.core import ContractAttributes, ContractType, ContractRole, ActusDateTime
from jactus.observers import ConstantRiskFactorObserver

def compute_delta(attrs, base_price=100.0, bump=1.0):
    """Compute delta via finite difference on the underlying price."""
    def pv_at_price(price):
        rf = ConstantRiskFactorObserver(constant_value=price)
        result = create_contract(attrs, rf).simulate()
        return sum(e.payoff for e in result.events)

    pv_base = pv_at_price(base_price)
    pv_bump = pv_at_price(base_price + bump)
    return (pv_bump - pv_base) / bump

results = []
with open("options.csv") as f:
    reader = csv.DictReader(f)
    for row in reader:
        attrs = ContractAttributes(
            contract_id=row["option_id"],
            contract_type=ContractType.OPTNS,
            contract_role=ContractRole.BUY,
            status_date=ActusDateTime(2024, 1, 1),
            initial_exchange_date=ActusDateTime(2024, 1, 15),
            maturity_date=ActusDateTime(
                int(row["expiry_year"]), int(row["expiry_month"]), int(row["expiry_day"])
            ),
            notional_principal=float(row["notional"]),
            contract_structure=f'{{"Underlier": "{row["underlier"]}"}}',
            option_type=row["option_type"],
            option_strike_1=float(row["strike"]),
            option_exercise_type=row.get("exercise_type", "E"),
        )

        delta = compute_delta(attrs, float(row["current_price"]))

        results.append([
            row["option_id"], row["option_type"], row["strike"],
            row["current_price"], f"{delta:.6f}",
        ])

# Write CSV
with open("greeks.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Option ID", "Type", "Strike", "Spot", "Delta"])
    writer.writerows(results)
```

3. **Write greeks to Sheets**

```bash
gws sheets create --title "Options Greeks Dashboard"
gws sheets import --spreadsheet-id "NEW_SHEET_ID" --file greeks.csv --sheet "Greeks"
```

### Expected Output

A Google Sheet with columns: Option ID, Type (C/P), Strike, Spot Price, Delta.

---

## Recipe 5: Calendar-Triggered Maturity Alert

Read Google Calendar for upcoming contract maturity dates, simulate PAM contracts
to check final payoff amounts, and send a Gmail alert if the payoff exceeds a threshold.

### Steps

1. **Query upcoming maturity events from Calendar**

```bash
gws calendar events list \
  --calendar-id "primary" \
  --time-min "$(date -I)" \
  --time-max "$(date -d '+30 days' -I)" \
  --query "contract maturity" \
  --format json > upcoming_maturities.json
```

2. **Simulate contracts and check thresholds**

```python
import json
from jactus.contracts import create_contract
from jactus.core import ContractAttributes, ContractType, ContractRole, ActusDateTime
from jactus.observers import ConstantRiskFactorObserver

ALERT_THRESHOLD = 1_000_000.0  # Alert if payoff > $1M

with open("upcoming_maturities.json") as f:
    events = json.load(f)

alerts = []
for event in events:
    # Parse contract details from calendar event description
    desc = event.get("description", "")
    contract_id = event.get("summary", "UNKNOWN")

    # Example: extract notional and rate from structured description
    # In practice, the agent would parse these from the event body
    notional = float(desc.split("notional:")[1].split()[0]) if "notional:" in desc else 100_000.0
    rate = float(desc.split("rate:")[1].split()[0]) if "rate:" in desc else 0.05

    # Parse maturity date from calendar event
    maturity_str = event["end"]["date"]  # YYYY-MM-DD
    y, m, d = map(int, maturity_str.split("-"))

    attrs = ContractAttributes(
        contract_id=contract_id,
        contract_type=ContractType.PAM,
        contract_role=ContractRole.RPA,
        status_date=ActusDateTime(2024, 1, 1),
        initial_exchange_date=ActusDateTime(2024, 1, 1),
        maturity_date=ActusDateTime(y, m, d),
        notional_principal=notional,
        nominal_interest_rate=rate,
    )

    rf = ConstantRiskFactorObserver(constant_value=0.0)
    result = create_contract(attrs, rf).simulate()

    # Check maturity event payoff
    md_events = [e for e in result.events if e.event_type.name == "MD"]
    if md_events:
        payoff = abs(md_events[0].payoff)
        if payoff > ALERT_THRESHOLD:
            alerts.append({
                "contract_id": contract_id,
                "maturity_date": maturity_str,
                "payoff": f"{payoff:,.2f}",
            })

# Build alert message
if alerts:
    body = "MATURITY ALERT: The following contracts have payoffs exceeding "
    body += f"${ALERT_THRESHOLD:,.0f}:\n\n"
    for a in alerts:
        body += f"  - {a['contract_id']}: matures {a['maturity_date']}, "
        body += f"payoff ${a['payoff']}\n"

    with open("maturity_alert.txt", "w") as f:
        f.write(body)
```

3. **Send alert email if thresholds exceeded**

```bash
if [ -f maturity_alert.txt ]; then
    gws gmail send \
      --to "treasury@example.com" \
      --subject "ALERT: Large Contract Maturities in Next 30 Days" \
      --body-file maturity_alert.txt
fi
```

### Expected Output

An email listing contracts maturing within 30 days whose payoff exceeds the
configured threshold, with contract ID, maturity date, and payoff amount.
