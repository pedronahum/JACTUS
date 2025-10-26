"""EUR-USD FX Swap Example (1 Year)

This example demonstrates a foreign exchange (FX) swap where:
- Two parties exchange currencies at spot rate
- Agreement to reverse the exchange at a future date (forward rate)
- Notional: EUR 1,000,000 / USD 1,100,000
- Spot rate: 1.10 USD/EUR
- Forward rate: 1.12 USD/EUR (implicit forward premium)
- Maturity: 1 year

FX swaps are used for:
1. Hedging currency exposure
2. Obtaining foreign currency funding
3. Managing liquidity across currencies
4. Arbitrage opportunities

This demonstrates the FXOUT (Foreign Exchange Outright) contract type.
"""

from jactus.contracts import create_contract
from jactus.core import (
    ActusDateTime,
    ContractAttributes,
    ContractRole,
    ContractType,
    DayCountConvention,
)
from jactus.observers import ConstantRiskFactorObserver


def main():
    """Run EUR-USD FX Swap example."""
    print("=" * 80)
    print("EUR-USD FOREIGN EXCHANGE SWAP (1 YEAR)")
    print("=" * 80)
    print()

    # ==================== Contract Setup ====================

    print("Contract Parameters:")
    print("-" * 80)

    # Contract dates
    status_date = ActusDateTime(2024, 1, 1, 0, 0, 0)
    maturity_date = ActusDateTime(2025, 1, 1, 0, 0, 0)
    settlement_date = maturity_date  # Forward settlement

    print(f"Trade Date:          {status_date.year}-{status_date.month:02d}-{status_date.day:02d}")
    print(
        f"Settlement Date:     {settlement_date.year}-{settlement_date.month:02d}-{settlement_date.day:02d}"
    )
    print(
        f"Maturity Date:       {maturity_date.year}-{maturity_date.month:02d}-{maturity_date.day:02d}"
    )
    print(f"Tenor:               1 year")
    print()

    # FX Swap terms
    eur_notional = 1_000_000.0  # EUR 1 million
    usd_notional = 1_100_000.0  # USD 1.1 million (at spot)
    spot_rate = 1.10  # 1 EUR = 1.10 USD
    forward_rate = 1.12  # 1 EUR = 1.12 USD (1 year forward)

    print("Currency Pair:       EUR/USD")
    print(f"EUR Notional:        EUR {eur_notional:,.0f}")
    print(f"USD Notional:        USD {usd_notional:,.0f}")
    print(f"Spot Rate:           {spot_rate:.4f} USD/EUR")
    print(f"Forward Rate:        {forward_rate:.4f} USD/EUR")
    print(f"Forward Points:      {(forward_rate - spot_rate) * 10000:.1f} pips")
    print()

    # ==================== FX Swap Mechanics ====================

    print("FX Swap Mechanics:")
    print("-" * 80)
    print()
    print("An FX swap consists of two legs:")
    print("1. Near Leg (Spot):  Exchange EUR for USD at spot rate")
    print("2. Far Leg (Forward): Reverse exchange at forward rate")
    print()
    print("In this example:")
    print(f"  Near Leg:  Receive EUR {eur_notional:,.0f}, Pay USD {usd_notional:,.0f}")
    print(f"             Rate: {spot_rate:.4f}")
    print()
    print(
        f"  Far Leg:   Pay EUR {eur_notional:,.0f}, Receive USD {eur_notional * forward_rate:,.0f}"
    )
    print(f"             Rate: {forward_rate:.4f}")
    print()
    print(f"  Net USD:   Receive USD {eur_notional * (forward_rate - spot_rate):,.0f}")
    print("             (Forward premium earned)")
    print()

    # ==================== Contract Creation ====================

    # Create FXOUT contract
    # We model the forward leg (far leg) of the swap
    # The contract holder will receive USD and pay EUR at maturity

    attrs = ContractAttributes(
        contract_id="FXSWAP-001",
        contract_type=ContractType.FXOUT,
        contract_role=ContractRole.RPA,  # Receive EUR (primary), pay USD (secondary)
        status_date=status_date,
        maturity_date=maturity_date,
        settlement_date=settlement_date,
        # Currency pair
        currency="EUR",  # Primary currency
        currency_2="USD",  # Secondary currency
        # Notional amounts
        notional_principal=eur_notional,  # EUR amount
        notional_principal_2=usd_notional,  # USD amount at spot
        # FX rate
        market_object_code_of_rate_reset="FX-EURUSD",
        # Settlement
        delivery_settlement="S",  # Gross settlement (both currencies exchanged)
        purchase_date=status_date,
        price_at_purchase_date=spot_rate,
        day_count_convention=DayCountConvention.AA,
    )

    # Risk factor observer with forward FX rate
    # The forward rate will be used at settlement
    rf_observer = ConstantRiskFactorObserver(forward_rate)

    print("Creating FX Swap contract...")
    fx_contract = create_contract(attrs, rf_observer)
    print(f"✓ Contract created: {fx_contract.attributes.contract_id}")
    print(f"  Contract Type: {fx_contract.attributes.contract_type.value}")
    print(f"  Role: {fx_contract.attributes.contract_role.value}")
    print(f"  Currency Pair: {fx_contract.attributes.currency}/{fx_contract.attributes.currency_2}")
    print()

    # ==================== State Initialization ====================

    print("Initializing contract state...")
    initial_state = fx_contract.initialize_state()
    print(f"✓ Initial state created")
    print(
        f"  Status Date (SD):    {initial_state.sd.year}-{initial_state.sd.month:02d}-{initial_state.sd.day:02d}"
    )
    print(
        f"  Maturity (TMD):      {initial_state.tmd.year}-{initial_state.tmd.month:02d}-{initial_state.tmd.day:02d}"
    )
    print(f"  EUR Notional (NT):   EUR {float(initial_state.nt):,.2f}")
    print(f"  Performance (PRF):   {initial_state.prf.value}")
    print()

    # ==================== Event Schedule ====================

    print("Generating event schedule...")
    schedule = fx_contract.generate_event_schedule()
    print(f"✓ Schedule generated: {len(schedule.events)} events")
    print()

    # Display events
    print("Event Schedule:")
    print("-" * 80)
    print(f"{'Event Type':<15} {'Date':<12} {'Payment (EUR)':<20} {'Description':<30}")
    print("-" * 80)

    for event in schedule.events:
        event_date = (
            f"{event.event_time.year}-{event.event_time.month:02d}-{event.event_time.day:02d}"
        )
        payoff_str = f"EUR {event.payoff:,.2f}" if event.payoff != 0 else "-"

        # Event descriptions
        descriptions = {
            "PRD": "Purchase (Spot Leg)",
            "STD": "Settlement (Forward Leg)",
            "MD": "Maturity",
        }
        desc = descriptions.get(event.event_type.value, "")

        print(f"{event.event_type.value:<15} {event_date:<12} {payoff_str:<20} {desc:<30}")

    print()

    # ==================== Simulation ====================

    print("Running simulation...")
    result = fx_contract.simulate(rf_observer)
    print(f"✓ Simulation complete")
    print()

    # ==================== Results Analysis ====================

    print("Cashflow Analysis:")
    print("-" * 80)
    print()

    # Calculate settlement amounts
    eur_settlement = -eur_notional  # Pay EUR
    usd_settlement = eur_notional * forward_rate  # Receive USD

    print("Forward Leg Settlement (at maturity):")
    print(f"  Pay:     EUR {abs(eur_settlement):,.2f}")
    print(f"  Receive: USD {usd_settlement:,.2f}")
    print(f"  Rate:    {forward_rate:.4f} USD/EUR")
    print()

    # Forward premium/discount
    forward_premium_usd = eur_notional * (forward_rate - spot_rate)
    forward_premium_bps = ((forward_rate / spot_rate) - 1) * 10000

    print("Forward Premium/Discount:")
    print(f"  Spot Rate:      {spot_rate:.4f}")
    print(f"  Forward Rate:   {forward_rate:.4f}")
    print(f"  Premium:        USD {forward_premium_usd:,.2f}")
    print(f"  Premium (bps):  {forward_premium_bps:.2f} basis points")
    print()

    # ==================== Valuation ====================

    print("FX Swap Valuation:")
    print("-" * 80)
    print()

    # The FX swap value is the difference between spot and forward legs
    # At inception, this should be close to zero (or equal to forward points)
    spot_leg_usd = eur_notional * spot_rate
    forward_leg_usd = eur_notional * forward_rate
    swap_value_usd = forward_leg_usd - spot_leg_usd

    print(f"Spot Leg Value:      USD {spot_leg_usd:,.2f}")
    print(f"Forward Leg Value:   USD {forward_leg_usd:,.2f}")
    print(f"Swap Value:          USD {swap_value_usd:,.2f}")
    print()

    # Interest rate differential (covered interest parity)
    # Forward premium reflects interest rate differential
    implied_usd_eur_spread = ((forward_rate / spot_rate) - 1) * 100
    print("Implied Interest Rate Differential:")
    print(f"  USD rate - EUR rate ≈ {implied_usd_eur_spread:.2f}% p.a.")
    print()
    print("Note: In equilibrium, forward premium/discount reflects interest")
    print("      rate differential between two currencies (covered interest parity)")
    print()

    # ==================== Market Scenarios ====================

    print("FX Rate Scenario Analysis:")
    print("-" * 80)
    print()

    scenarios = [
        (1.08, "EUR Appreciation (USD Depreciation)"),
        (1.10, "Spot Rate (No Change)"),
        (1.12, "Forward Rate (Par)"),
        (1.14, "EUR Depreciation (USD Appreciation)"),
    ]

    print(f"{'Scenario':<40} {'Rate':<12} {'USD Settlement':<20} {'P&L (USD)':<15}")
    print("-" * 80)

    for rate, scenario_name in scenarios:
        usd_received = eur_notional * rate
        pnl = usd_received - usd_notional
        sign = "+" if pnl > 0 else ""

        print(
            f"{scenario_name:<40} {rate:>6.4f}     USD {usd_received:>13,.2f}  {sign}{pnl:>13,.2f}"
        )

    print()
    print("Note: P&L calculated relative to initial USD notional at spot rate")
    print()

    # ==================== Use Cases ====================

    print("Common Use Cases for FX Swaps:")
    print("-" * 80)
    print()
    print("1. Hedging Currency Exposure:")
    print("   - Lock in future exchange rate for known cash flows")
    print("   - Eliminate FX risk on foreign currency positions")
    print()
    print("2. Foreign Currency Funding:")
    print("   - Obtain foreign currency without spot market execution")
    print("   - Cheaper than borrowing in foreign market")
    print()
    print("3. Liquidity Management:")
    print("   - Temporarily swap currencies for liquidity needs")
    print("   - Manage cash across different currency zones")
    print()
    print("4. Arbitrage:")
    print("   - Exploit deviations from covered interest parity")
    print("   - Cross-currency basis trading")
    print()
    print("5. Rolling FX Positions:")
    print("   - Extend or reduce FX exposure without closing position")
    print("   - Common in carry trade strategies")
    print()

    # ==================== Key Points ====================

    print("Key Points:")
    print("-" * 80)
    print()
    print("• FX swaps have TWO legs: near (spot) and far (forward)")
    print("• Forward rate includes interest rate differential (covered interest parity)")
    print("• No principal exchange at inception (unlike cross-currency swaps)")
    print("• Widely used by corporates, banks, and central banks")
    print("• Largest segment of FX market (>$3 trillion daily volume)")
    print("• Typically short-dated (overnight to 1 year)")
    print()

    print("=" * 80)
    print("EXAMPLE COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
