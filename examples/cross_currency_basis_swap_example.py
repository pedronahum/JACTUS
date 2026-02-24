"""Cross-Currency Basis Swap Example (5 Years)

This example demonstrates a cross-currency basis swap where:
- Two parties exchange notional principals in different currencies
- Each party pays floating interest in their respective currency
- Floating rates: 3-month EURIBOR (EUR) vs 3-month SOFR (USD)
- Basis spread: +30 bps on USD leg
- Notional: EUR 10,000,000 / USD 11,000,000
- Exchange rate: 1.10 USD/EUR
- Maturity: 5 years

Cross-currency basis swaps are used for:
1. Hedging multi-currency cash flows
2. Accessing foreign currency funding markets
3. Managing long-term FX exposure
4. Arbitrage opportunities (basis trading)

This demonstrates the SWAPS (Swap) contract type with composition.
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
from jactus.observers import ConstantRiskFactorObserver, MockChildContractObserver


def main():
    """Run cross-currency basis swap example."""
    print("=" * 80)
    print("CROSS-CURRENCY BASIS SWAP: EUR vs USD (5 YEARS)")
    print("=" * 80)
    print()

    # ==================== Contract Setup ====================

    print("Contract Parameters:")
    print("-" * 80)

    # Contract dates
    status_date = ActusDateTime(2024, 1, 1, 0, 0, 0)
    maturity_date = ActusDateTime(2029, 1, 1, 0, 0, 0)

    print(f"Trade Date:          {status_date.year}-{status_date.month:02d}-{status_date.day:02d}")
    print(
        f"Maturity Date:       {maturity_date.year}-{maturity_date.month:02d}-{maturity_date.day:02d}"
    )
    print(f"Tenor:               5 years")
    print()

    # Swap terms
    eur_notional = 10_000_000.0  # EUR 10 million
    usd_notional = 11_000_000.0  # USD 11 million
    fx_rate = 1.10  # 1 EUR = 1.10 USD
    euribor_3m = 0.035  # 3.5% EURIBOR
    sofr_3m = 0.045  # 4.5% SOFR
    basis_spread = 0.0030  # 30 bps basis spread on USD leg

    print("Notional Amounts:")
    print(f"  EUR Leg:           EUR {eur_notional:,.0f}")
    print(f"  USD Leg:           USD {usd_notional:,.0f}")
    print(f"  Exchange Rate:     {fx_rate:.4f} USD/EUR")
    print()
    print("Floating Rates:")
    print(f"  EUR Leg:           3M EURIBOR ({euribor_3m * 100:.2f}%)")
    print(
        f"  USD Leg:           3M SOFR ({sofr_3m * 100:.2f}%) + {basis_spread * 10000:.0f} bps basis"
    )
    print(f"  Effective USD:     {(sofr_3m + basis_spread) * 100:.2f}%")
    print()
    print("Payment Frequency:   Quarterly (3 months)")
    print("Day Count:           Actual/360")
    print()

    # ==================== Cross-Currency Swap Mechanics ====================

    print("Cross-Currency Basis Swap Mechanics:")
    print("-" * 80)
    print()
    print("A cross-currency basis swap has the following cashflows:")
    print()
    print("1. INITIAL EXCHANGE (at inception):")
    print(f"   Party A: Pays USD {usd_notional:,.0f}")
    print(f"   Party A: Receives EUR {eur_notional:,.0f}")
    print()
    print("2. PERIODIC INTEREST PAYMENTS (quarterly):")
    print(f"   Party A: Pays EUR interest (EURIBOR on EUR {eur_notional:,.0f})")
    print(f"   Party A: Receives USD interest (SOFR + basis on USD {usd_notional:,.0f})")
    print()
    print("3. FINAL EXCHANGE (at maturity):")
    print(f"   Party A: Receives USD {usd_notional:,.0f}")
    print(f"   Party A: Pays EUR {eur_notional:,.0f}")
    print()
    print("Net Effect: Party A has effectively borrowed EUR and lent USD")
    print()

    # ==================== Child Contract Creation ====================

    # For SWAPS, we need to create child contracts for each leg
    # and use a MockChildContractObserver to track them

    print("Creating child leg contracts...")
    print()

    # EUR Leg (Party A pays EUR floating)
    eur_leg_attrs = ContractAttributes(
        contract_id="EUR-LEG-001",
        contract_type=ContractType.SWPPV,
        contract_role=ContractRole.RFL,  # Pays floating
        status_date=status_date,
        maturity_date=maturity_date,
        notional_principal=eur_notional,
        nominal_interest_rate=euribor_3m,
        nominal_interest_rate_2=euribor_3m,
        interest_payment_cycle="3M",
        interest_payment_anchor=status_date,
        rate_reset_cycle="3M",
        rate_reset_anchor=status_date,
        delivery_settlement="D",  # Net
        day_count_convention=DayCountConvention.AA,
        end_of_month_convention=EndOfMonthConvention.SD,
        currency="EUR",
    )

    # USD Leg (Party A receives USD floating + basis)
    usd_leg_attrs = ContractAttributes(
        contract_id="USD-LEG-001",
        contract_type=ContractType.SWPPV,
        contract_role=ContractRole.RPA,  # Receives floating
        status_date=status_date,
        maturity_date=maturity_date,
        notional_principal=usd_notional,
        nominal_interest_rate=sofr_3m + basis_spread,
        nominal_interest_rate_2=sofr_3m + basis_spread,
        interest_payment_cycle="3M",
        interest_payment_anchor=status_date,
        rate_reset_cycle="3M",
        rate_reset_anchor=status_date,
        delivery_settlement="D",  # Net
        day_count_convention=DayCountConvention.AA,
        end_of_month_convention=EndOfMonthConvention.SD,
        currency="USD",
    )

    # Create risk factor observers
    eur_rf_observer = ConstantRiskFactorObserver(euribor_3m)
    usd_rf_observer = ConstantRiskFactorObserver(sofr_3m + basis_spread)

    # Create child contracts
    eur_leg = create_contract(eur_leg_attrs, eur_rf_observer)
    usd_leg = create_contract(usd_leg_attrs, usd_rf_observer)

    print(f"✓ EUR Leg created: {eur_leg.attributes.contract_id}")
    print(f"  Notional: EUR {eur_notional:,.0f}")
    print(f"  Rate: {euribor_3m * 100:.2f}% (3M EURIBOR)")
    print()
    print(f"✓ USD Leg created: {usd_leg.attributes.contract_id}")
    print(f"  Notional: USD {usd_notional:,.0f}")
    print(
        f"  Rate: {(sofr_3m + basis_spread) * 100:.2f}% (3M SOFR + {basis_spread * 10000:.0f} bps)"
    )
    print()

    # ==================== Parent SWAPS Contract ====================

    # Create child contract observer
    child_observer = MockChildContractObserver()

    # Register child contracts
    # For cross-currency swap, we need to register states and events for both legs
    eur_state = eur_leg.initialize_state()
    usd_state = usd_leg.initialize_state()

    eur_schedule = eur_leg.generate_event_schedule()
    usd_schedule = usd_leg.generate_event_schedule()

    child_observer.register_child(
        "EUR-LEG-001",
        events=list(eur_schedule.events),
        state=eur_state,
        attributes={
            "notional_principal": eur_notional,
            "nominal_interest_rate": euribor_3m,
        },
    )

    child_observer.register_child(
        "USD-LEG-001",
        events=list(usd_schedule.events),
        state=usd_state,
        attributes={
            "notional_principal": usd_notional,
            "nominal_interest_rate": sofr_3m + basis_spread,
        },
    )

    # Create SWAPS contract with child references
    swaps_attrs = ContractAttributes(
        contract_id="XCCY-SWAP-001",
        contract_type=ContractType.SWAPS,
        contract_role=ContractRole.RPA,  # Receives USD (asset), pays EUR
        status_date=status_date,
        maturity_date=maturity_date,
        # Reference to child legs
        contract_structure='{"FirstLeg": "EUR-LEG-001", "SecondLeg": "USD-LEG-001"}',
        # Settlement
        delivery_settlement="D",  # Net settlement of interest
        currency="USD",  # Reporting currency
    )

    rf_observer = ConstantRiskFactorObserver(fx_rate)

    print("Creating cross-currency basis swap contract...")
    xccy_swap = create_contract(swaps_attrs, rf_observer, child_observer)
    print(f"✓ Contract created: {xccy_swap.attributes.contract_id}")
    print(f"  Contract Type: {xccy_swap.attributes.contract_type.value}")
    print(f"  Role: {xccy_swap.attributes.contract_role.value}")
    print(f"  EUR Leg: {eur_leg_attrs.contract_id}")
    print(f"  USD Leg: {usd_leg_attrs.contract_id}")
    print()

    # ==================== State Initialization ====================

    print("Initializing contract state...")
    initial_state = xccy_swap.initialize_state()
    print(f"✓ Initial state created")
    print(
        f"  Status Date (SD):    {initial_state.sd.year}-{initial_state.sd.month:02d}-{initial_state.sd.day:02d}"
    )
    print(
        f"  Maturity (TMD):      {initial_state.tmd.year}-{initial_state.tmd.month:02d}-{initial_state.tmd.day:02d}"
    )
    print(f"  Performance (PRF):   {initial_state.prf.value}")
    print()

    # ==================== Event Schedule ====================

    print("Generating event schedule...")
    schedule = xccy_swap.generate_event_schedule()
    print(f"✓ Schedule generated: {len(schedule.events)} events")
    print()

    # Display first few events
    print("Event Schedule (first 15 events):")
    print("-" * 80)
    print(f"{'Event Type':<15} {'Date':<12} {'Description':<50}")
    print("-" * 80)

    for i, event in enumerate(schedule.events[:15]):
        event_date = (
            f"{event.event_time.year}-{event.event_time.month:02d}-{event.event_time.day:02d}"
        )

        # Event descriptions
        descriptions = {
            "IED": "Initial Exchange (principal swap)",
            "IP": "Interest Payment (net interest)",
            "IPFX": "Fixed Interest Payment",
            "IPFL": "Floating Interest Payment",
            "PR": "Principal Repayment (final exchange)",
            "MD": "Maturity",
        }
        desc = descriptions.get(event.event_type.value, event.event_type.value)

        print(f"{event.event_type.value:<15} {event_date:<12} {desc:<50}")

    if len(schedule.events) > 15:
        print(f"... and {len(schedule.events) - 15} more events")
    print()

    # ==================== Simulation ====================

    print("Running simulation...")
    result = xccy_swap.simulate(rf_observer, child_observer)
    print(f"✓ Simulation complete")
    print()

    # ==================== Cashflow Analysis ====================

    print("Cashflow Analysis:")
    print("-" * 80)
    print()

    # Calculate quarterly interest payments
    eur_quarterly_payment = eur_notional * euribor_3m / 4
    usd_quarterly_payment = usd_notional * (sofr_3m + basis_spread) / 4

    print("Quarterly Interest Payments:")
    print(f"  EUR Leg (Pay):       EUR {eur_quarterly_payment:,.2f}")
    print(f"  USD Leg (Receive):   USD {usd_quarterly_payment:,.2f}")
    print()

    # Net interest (converted to USD for comparison)
    eur_payment_in_usd = eur_quarterly_payment * fx_rate
    net_interest_usd = usd_quarterly_payment - eur_payment_in_usd

    print("Net Quarterly Interest (in USD):")
    print(f"  Receive USD:         USD {usd_quarterly_payment:,.2f}")
    print(f"  Pay EUR (in USD):    USD {eur_payment_in_usd:,.2f}")
    print(f"  Net:                 USD {net_interest_usd:,.2f}")
    print()

    # Total over 5 years
    total_periods = 20  # Quarterly for 5 years
    total_net_interest = net_interest_usd * total_periods

    print(f"Total Net Interest over 5 years: USD {total_net_interest:,.2f}")
    print()

    # ==================== Basis Spread Impact ====================

    print("Basis Spread Impact:")
    print("-" * 80)
    print()

    # Interest without basis spread
    usd_payment_no_basis = usd_notional * sofr_3m / 4
    basis_impact_quarterly = usd_notional * basis_spread / 4
    basis_impact_total = basis_impact_quarterly * total_periods

    print(f"USD Interest without basis:  USD {usd_payment_no_basis:,.2f} per quarter")
    print(f"Basis spread contribution:   USD {basis_impact_quarterly:,.2f} per quarter")
    print(f"Total basis over 5 years:    USD {basis_impact_total:,.2f}")
    print()
    print(
        f"The {basis_spread * 10000:.0f} bps basis spread adds USD {basis_impact_total:,.0f} to USD receipts"
    )
    print()

    # ==================== Market Scenarios ====================

    print("Basis Spread Scenario Analysis:")
    print("-" * 80)
    print()

    scenarios = [
        (-0.0010, "Negative Basis (EUR strength)"),
        (0.0000, "Zero Basis"),
        (0.0030, "Current Basis (+30 bps)"),
        (0.0050, "Wide Basis (+50 bps)"),
    ]

    print(f"{'Scenario':<40} {'Basis':<12} {'Quarterly USD':<20} {'5Y Total USD':<15}")
    print("-" * 80)

    for basis, scenario_name in scenarios:
        usd_payment = usd_notional * (sofr_3m + basis) / 4
        total_5y = usd_payment * total_periods

        print(
            f"{scenario_name:<40} {basis * 10000:>6.0f} bps   USD {usd_payment:>13,.2f}  USD {total_5y:>13,.0f}"
        )

    print()

    # ==================== FX Risk ====================

    print("FX Risk Analysis:")
    print("-" * 80)
    print()
    print("Cross-currency swaps are exposed to FX rate changes:")
    print()

    fx_scenarios = [
        (1.05, "EUR Appreciation"),
        (1.10, "Current Rate"),
        (1.15, "EUR Depreciation"),
    ]

    print(f"{'Scenario':<30} {'FX Rate':<15} {'EUR Payment (USD)':<20} {'Net Interest (USD)':<20}")
    print("-" * 80)

    for fx, scenario_name in fx_scenarios:
        eur_in_usd = eur_quarterly_payment * fx
        net = usd_quarterly_payment - eur_in_usd
        sign = "+" if net > 0 else ""

        print(
            f"{scenario_name:<30} {fx:>8.4f}        USD {eur_in_usd:>13,.2f}  {sign}USD {net:>13,.2f}"
        )

    print()
    print("Note: FX rate changes affect the USD value of EUR payments,")
    print("      changing the net interest in USD terms")
    print()

    # ==================== Use Cases ====================

    print("Common Use Cases for Cross-Currency Basis Swaps:")
    print("-" * 80)
    print()
    print("1. Multi-Currency Debt Management:")
    print("   - Company with EUR revenue hedges USD debt")
    print("   - Converts USD borrowing to synthetic EUR borrowing")
    print()
    print("2. Foreign Subsidiary Funding:")
    print("   - Parent company provides USD to foreign EUR subsidiary")
    print("   - Hedge long-term FX exposure of subsidiary operations")
    print()
    print("3. Asset-Liability Matching:")
    print("   - Match currency of assets and liabilities")
    print("   - Pension funds and insurance companies use extensively")
    print()
    print("4. Basis Trading:")
    print("   - Exploit cross-currency basis anomalies")
    print("   - Post-2008 crisis, basis became persistently non-zero")
    print()
    print("5. Central Bank Operations:")
    print("   - Central banks use XCCY swaps for FX reserves management")
    print("   - Fed USD liquidity swaps with other central banks")
    print()

    # ==================== Key Differences ====================

    print("Cross-Currency Basis Swap vs Other Swaps:")
    print("-" * 80)
    print()
    print("│ Feature                 │ Interest Rate Swap  │ FX Swap      │ XCCY Basis Swap │")
    print("├─────────────────────────┼────────────────────┼──────────────┼─────────────────┤")
    print("│ Principal Exchange      │ NO                 │ YES (2x)     │ YES (2x)        │")
    print("│ Interest Payments       │ YES (1 currency)   │ NO           │ YES (2 currency)│")
    print("│ Floating Rates          │ Both legs          │ N/A          │ Both legs       │")
    print("│ FX Risk                 │ NO                 │ Fixed rate   │ Fixed rate      │")
    print("│ Typical Maturity        │ 2-10 years         │ 1 day - 1yr  │ 1-30 years      │")
    print("│ Basis Spread            │ N/A                │ Implicit     │ Explicit        │")
    print()

    # ==================== Market Context ====================

    print("Market Context:")
    print("-" * 80)
    print()
    print("Cross-Currency Basis Spread:")
    print("  • Before 2008: basis was close to zero (CIP held)")
    print("  • Post-2008: persistent non-zero basis (CIP breakdown)")
    print("  • Reflects funding costs and regulatory capital requirements")
    print("  • EUR/USD basis typically negative (EUR funding premium)")
    print("  • Basis can be 30-100+ bps depending on market stress")
    print()
    print("In this example:")
    print(f"  • USD receives {basis_spread * 10000:.0f} bps basis")
    print("  • Reflects relative cost of funding in each currency")
    print("  • Typical for EUR/USD where USD funding is premium")
    print()

    print("=" * 80)
    print("EXAMPLE COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
