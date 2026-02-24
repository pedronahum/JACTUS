"""Interest Rate Swap Example: Overnight Floating Leg vs Fixed Leg

This example demonstrates a plain vanilla interest rate swap where:
- Party A pays a fixed rate (5.0% annual)
- Party B pays a floating overnight (O/N) rate
- Notional: $10,000,000
- Maturity: 5 years
- Payment frequency: Quarterly

The swap uses net settlement where only the difference between fixed
and floating payments is exchanged.

This demonstrates the SWPPV (Plain Vanilla Swap) contract type.
"""

from jactus.contracts import create_contract
from jactus.core import (
    ActusDateTime,
    ContractAttributes,
    ContractRole,
    ContractType,
    DayCountConvention,
    EndOfMonthConvention,
)
from jactus.observers import ConstantRiskFactorObserver


def main():
    """Run interest rate swap example."""
    print("=" * 80)
    print("INTEREST RATE SWAP: O/N FLOATING LEG vs FIXED LEG")
    print("=" * 80)
    print()

    # ==================== Contract Setup ====================

    print("Contract Parameters:")
    print("-" * 80)

    # Contract dates
    status_date = ActusDateTime(2024, 1, 1, 0, 0, 0)
    maturity_date = ActusDateTime(2029, 1, 1, 0, 0, 0)

    print(f"Status Date:         {status_date.year}-{status_date.month:02d}-{status_date.day:02d}")
    print(
        f"Maturity Date:       {maturity_date.year}-{maturity_date.month:02d}-{maturity_date.day:02d}"
    )
    print("Tenor:               5 years")
    print()

    # Swap terms
    notional = 10_000_000.0  # $10 million
    fixed_rate = 0.05  # 5.0% annual
    floating_rate_initial = 0.03  # 3.0% initial O/N rate (will vary)

    print(f"Notional:            ${notional:,.0f}")
    print(f"Fixed Rate:          {fixed_rate * 100:.2f}% p.a.")
    print(f"Floating Rate:       O/N rate (initial {floating_rate_initial * 100:.2f}%)")
    print("Payment Frequency:   Quarterly")
    print("Day Count:           Actual/360")
    print()

    # ==================== Contract Creation ====================

    # Create SWPPV contract (Plain Vanilla Swap)
    # In SWPPV:
    # - RPA (Real Position Asset) = receives fixed, pays floating
    # - RPL (Real Position Liability) = pays fixed, receives floating
    #
    # We'll model from the fixed rate payer's perspective (RPL)

    attrs = ContractAttributes(
        contract_id="IRS-001",
        contract_type=ContractType.SWPPV,
        contract_role=ContractRole.RFL,  # Pays fixed (reference leg)
        status_date=status_date,
        maturity_date=maturity_date,
        notional_principal=notional,
        # Fixed leg (reference leg)
        nominal_interest_rate=fixed_rate,
        day_count_convention=DayCountConvention.AA,  # Actual/360
        # Floating leg (variable leg 2)
        nominal_interest_rate_2=floating_rate_initial,
        # Payment cycles
        interest_payment_cycle="3M",  # Quarterly interest payments (every 3 months)
        interest_payment_anchor=status_date,
        # Rate reset cycle for floating leg (O/N = daily resets)
        # Using weekly resets for demonstration (daily would create 1825 events!)
        rate_reset_cycle="1W",  # Weekly resets for overnight rate
        rate_reset_anchor=status_date,
        # Settlement
        delivery_settlement="D",  # Net settlement (only difference is paid)
        end_of_month_convention=EndOfMonthConvention.SD,
        currency="USD",
    )

    # Risk factor observer with initial O/N rate
    rf_observer = ConstantRiskFactorObserver(floating_rate_initial)

    print("Creating Interest Rate Swap contract...")
    swap_contract = create_contract(attrs, rf_observer)
    print(f"✓ Contract created: {swap_contract.attributes.contract_id}")
    print(f"  Contract Type: {swap_contract.attributes.contract_type.value}")
    print(f"  Role: {swap_contract.attributes.contract_role.value} (Fixed Rate Payer)")
    print()

    # ==================== State Initialization ====================

    print("Initializing contract state...")
    initial_state = swap_contract.initialize_state()
    print("✓ Initial state created")
    print(
        f"  Status Date (SD):    {initial_state.sd.year}-{initial_state.sd.month:02d}-{initial_state.sd.day:02d}"
    )
    print(
        f"  Maturity (TMD):      {initial_state.tmd.year}-{initial_state.tmd.month:02d}-{initial_state.tmd.day:02d}"
    )
    print(f"  Notional (NT):       ${float(initial_state.nt):,.2f}")
    print(f"  Fixed Rate (IPNR):   {float(initial_state.ipnr) * 100:.2f}%")
    print(f"  Performance (PRF):   {initial_state.prf.value}")
    print()

    # ==================== Event Schedule ====================

    print("Generating event schedule...")
    schedule = swap_contract.generate_event_schedule()
    print(f"✓ Schedule generated: {len(schedule.events)} events")
    print()

    # Display events
    print("Event Schedule:")
    print("-" * 80)
    print(f"{'Event Type':<15} {'Date':<12} {'Payment':<20} {'Description':<30}")
    print("-" * 80)

    for event in schedule.events[:10]:  # Show first 10 events
        event_date = (
            f"{event.event_time.year}-{event.event_time.month:02d}-{event.event_time.day:02d}"
        )
        payoff_str = f"${event.payoff:,.2f}" if event.payoff != 0 else "-"

        # Event descriptions
        descriptions = {
            "IED": "Initial Exchange",
            "IP": "Interest Payment (Net)",
            "IPFX": "Fixed Leg Payment",
            "IPFL": "Floating Leg Payment",
            "MD": "Maturity",
        }
        desc = descriptions.get(event.event_type.value, "")

        print(f"{event.event_type.value:<15} {event_date:<12} {payoff_str:<20} {desc:<30}")

    if len(schedule.events) > 10:
        print(f"... and {len(schedule.events) - 10} more events")
    print()

    # ==================== Simulation ====================

    print("Running simulation...")
    swap_contract.simulate(rf_observer)
    print("✓ Simulation complete")
    print()

    # ==================== Results Analysis ====================

    print("Cashflow Analysis:")
    print("-" * 80)

    # Calculate net payments
    total_net_payments = 0.0

    # In SWPPV with net settlement, we get net cashflows
    # Positive = receive, Negative = pay (from contract holder's perspective)
    for event in schedule.events:
        if event.event_type.value in ["IP", "IPFX", "IPFL"]:
            # For RFL (pays fixed), negative payoff means paying fixed
            # positive payoff means receiving floating
            total_net_payments += event.payoff

    print(f"Total Net Payments:     ${total_net_payments:,.2f}")
    print()

    # ==================== Swap Valuation Summary ====================

    print("Swap Valuation Summary:")
    print("-" * 80)
    print(f"Notional Amount:        ${notional:,.0f}")
    print(f"Fixed Rate Leg:         {fixed_rate * 100:.2f}% p.a.")
    print(f"Floating Rate Leg:      O/N rate (currently {floating_rate_initial * 100:.2f}%)")
    print()
    print("Fixed Leg Cashflows:")
    print(f"  - Quarterly payments of approximately ${(notional * fixed_rate / 4):,.2f}")
    print(f"  - {(maturity_date.year - status_date.year) * 4} total payments over 5 years")
    print()
    print("Floating Leg Cashflows:")
    print("  - Quarterly payments based on O/N rate resets")
    print(f"  - Current quarterly payment ~${(notional * floating_rate_initial / 4):,.2f}")
    print()
    print("Net Settlement:")
    print("  - Only the difference between fixed and floating is exchanged")
    print(f"  - If O/N rate < {fixed_rate * 100:.2f}%, fixed payer receives")
    print(f"  - If O/N rate > {fixed_rate * 100:.2f}%, fixed payer pays")
    print()

    # ==================== Market Scenarios ====================

    print("Market Scenario Analysis:")
    print("-" * 80)
    print()

    scenarios = [
        (0.02, "Low Rate Environment"),
        (0.03, "Current O/N Rate"),
        (0.05, "Swap Rate (Par)"),
        (0.07, "High Rate Environment"),
    ]

    print(f"{'Scenario':<30} {'O/N Rate':<12} {'Quarterly Net Payment':<25}")
    print("-" * 80)

    for rate, scenario_name in scenarios:
        # Approximate quarterly payment difference
        fixed_payment = notional * fixed_rate / 4
        floating_payment = notional * rate / 4
        net_payment = floating_payment - fixed_payment  # From fixed payer's view

        sign = "+" if net_payment > 0 else ""
        print(f"{scenario_name:<30} {rate * 100:>6.2f}%     {sign}${net_payment:>18,.2f}")

    print()
    print("Note: Actual payments depend on day count conventions and compounding.")
    print()

    # ==================== Use Cases ====================

    print("Common Use Cases for Interest Rate Swaps:")
    print("-" * 80)
    print()
    print("1. Interest Rate Risk Management:")
    print("   - Company with floating-rate debt swaps to fixed rate")
    print("   - Protects against rising interest rates")
    print()
    print("2. Asset-Liability Matching:")
    print("   - Banks match fixed-rate assets with fixed-rate liabilities")
    print("   - Reduces duration mismatch risk")
    print()
    print("3. Speculation:")
    print("   - Take directional views on interest rate movements")
    print("   - Profit from rate differentials")
    print()
    print("4. Arbitrage:")
    print("   - Exploit pricing differences between swap and bond markets")
    print()

    print("=" * 80)
    print("EXAMPLE COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
