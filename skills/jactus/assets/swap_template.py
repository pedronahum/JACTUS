"""SWPPV (Plain Vanilla Interest Rate Swap) template.

Demonstrates a 5-year fixed/floating interest rate swap with net settlement
cash flows using the SWPPV single-contract model.
"""

from jactus.contracts import create_contract
from jactus.core import ContractAttributes, ContractType, ContractRole, ActusDateTime
from jactus.observers import ConstantRiskFactorObserver, TimeSeriesRiskFactorObserver


def create_swap_fixed_receiver():
    """Create a 5-year IRS: receive 4% fixed, pay 3M LIBOR floating.

    Notional: $10,000,000
    Fixed rate: 4.0%
    Initial floating rate: 3.5%
    Payment frequency: Semi-annual
    Rate reset: Quarterly
    """
    return ContractAttributes(
        contract_id="IRS-001",
        contract_type=ContractType.SWPPV,
        contract_role=ContractRole.RFL,  # Receive First Leg (fixed)
        status_date=ActusDateTime(2024, 1, 1),
        initial_exchange_date=ActusDateTime(2024, 1, 1),
        maturity_date=ActusDateTime(2029, 1, 1),
        notional_principal=10_000_000.0,
        nominal_interest_rate=0.04,        # Fixed leg rate
        nominal_interest_rate_2=0.035,     # Initial floating leg rate
        interest_payment_cycle="6M",       # Semi-annual payments
        rate_reset_cycle="3M",             # Quarterly resets
        rate_reset_market_object="LIBOR-3M",
    )


def simulate_flat_rate():
    """Simulate swap with a constant floating rate (no rate changes)."""
    attrs = create_swap_fixed_receiver()

    # Flat floating rate scenario: LIBOR stays at 3.5%
    rf = ConstantRiskFactorObserver(constant_value=0.035)
    contract = create_contract(attrs, rf)
    result = contract.simulate()

    print("=" * 70)
    print("  IRS-001: 5Y Fixed/Floating Swap — Flat Rate Scenario")
    print("  Receive 4.0% fixed, Pay LIBOR-3M (flat at 3.5%)")
    print("=" * 70)

    print(f"\n{'Date':<25} {'Event':<8} {'Payoff':>15}")
    print("-" * 50)

    total_fixed = 0.0
    total_floating = 0.0

    for event in result.events:
        if event.payoff != 0:
            print(f"{str(event.event_time):<25} {event.event_type.name:<8} {event.payoff:>15,.2f}")
            if event.event_type.name in ("IPFX", "IP"):
                total_fixed += event.payoff
            elif event.event_type.name == "IPFL":
                total_floating += event.payoff

    net = sum(e.payoff for e in result.events)
    print("-" * 50)
    print(f"{'Net Settlement':<33} {net:>15,.2f}")


def simulate_rising_rates():
    """Simulate swap with rising floating rates over the 5-year term."""
    attrs = create_swap_fixed_receiver()

    # Rising rate scenario
    rf = TimeSeriesRiskFactorObserver(
        time_series={
            "LIBOR-3M": [
                (ActusDateTime(2024, 1, 1), 0.035),
                (ActusDateTime(2024, 7, 1), 0.038),
                (ActusDateTime(2025, 1, 1), 0.042),
                (ActusDateTime(2025, 7, 1), 0.045),
                (ActusDateTime(2026, 1, 1), 0.048),
                (ActusDateTime(2026, 7, 1), 0.050),
                (ActusDateTime(2027, 1, 1), 0.052),
                (ActusDateTime(2028, 1, 1), 0.055),
                (ActusDateTime(2029, 1, 1), 0.055),
            ]
        },
        interpolation="step",
    )

    contract = create_contract(attrs, rf)
    result = contract.simulate()

    print("\n" + "=" * 70)
    print("  IRS-001: 5Y Fixed/Floating Swap — Rising Rate Scenario")
    print("  Receive 4.0% fixed, Pay LIBOR-3M (rising from 3.5% to 5.5%)")
    print("=" * 70)

    print(f"\n{'Date':<25} {'Event':<8} {'Payoff':>15}")
    print("-" * 50)

    for event in result.events:
        if event.payoff != 0:
            print(f"{str(event.event_time):<25} {event.event_type.name:<8} {event.payoff:>15,.2f}")

    net = sum(e.payoff for e in result.events)
    print("-" * 50)
    print(f"{'Net Settlement':<33} {net:>15,.2f}")
    print(f"\nNote: Negative net = payer benefits (floating rates rose above fixed)")


if __name__ == "__main__":
    simulate_flat_rate()
    simulate_rising_rates()
