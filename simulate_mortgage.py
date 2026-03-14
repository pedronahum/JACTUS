from jactus.contracts import create_contract
from jactus.core import ContractAttributes, ContractType, ContractRole, ActusDateTime, DayCountConvention
from jactus.observers import ConstantRiskFactorObserver
from collections import defaultdict

# 1. Define attributes for a 30-year mortgage
# $500k at 6.5% interest
attrs = ContractAttributes(
    contract_id="MORTGAGE-500K",
    contract_type=ContractType.ANN,
    contract_role=ContractRole.RPL,  # Real Position Liability (borrower)
    status_date=ActusDateTime(2026, 3, 5),
    initial_exchange_date=ActusDateTime(2026, 3, 5),
    maturity_date=ActusDateTime(2056, 3, 5),
    notional_principal=500_000.0,
    nominal_interest_rate=0.065,
    day_count_convention=DayCountConvention.A360,
    principal_redemption_cycle="1M",
    interest_payment_cycle="1M",
    principal_redemption_anchor=ActusDateTime(2026, 4, 5),
    interest_payment_anchor=ActusDateTime(2026, 4, 5),
)

# 2. Simulate
# For ANN, we can use a constant risk factor of 0 since it's a fixed rate
# (The nominal_interest_rate in attributes is used)
rf = ConstantRiskFactorObserver(constant_value=0.0)
contract = create_contract(attrs, rf)
result = contract.simulate()

# 3. Group events by date and format output
monthly_payments = defaultdict(lambda: {"principal": 0.0, "interest": 0.0, "total": 0.0})

for event in result.events:
    # We are the borrower (RPL), so payoffs are negative (outflows)
    # Convert to positive for display
    amount = abs(float(event.payoff))
    date_str = event.event_time.to_iso()[:10]
    
    if event.event_type == "PR":
        monthly_payments[date_str]["principal"] += amount
        monthly_payments[date_str]["total"] += amount
    elif event.event_type == "IP":
        monthly_payments[date_str]["interest"] += amount
        monthly_payments[date_str]["total"] += amount

# 4. Show first 12 months of payments (excluding IED which is at t=0)
sorted_dates = sorted(monthly_payments.keys())
# Skip the first date if it's IED (2026-03-05) and has no PR/IP
if sorted_dates and sorted_dates[0] == "2026-03-05" and monthly_payments[sorted_dates[0]]["total"] == 0:
    sorted_dates = sorted_dates[1:]

print(f"{'Month':<12} | {'Total Payment':<15} | {'Principal':<12} | {'Interest':<12}")
print("-" * 60)

count = 0
for date in sorted_dates:
    pay = monthly_payments[date]
    if pay["total"] > 0:
        print(f"{date:<12} | ${pay['total']:>13,.2f} | ${pay['principal']:>10,.2f} | ${pay['interest']:>10,.2f}")
        count += 1
        if count == 12:
            break
