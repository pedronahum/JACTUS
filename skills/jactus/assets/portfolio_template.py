"""Portfolio batch simulation template.

Demonstrates simulating a portfolio of PAM contracts using JACTUS's
portfolio API for batch execution.
"""

from jactus.contracts import create_contract
from jactus.core import ContractAttributes, ContractType, ContractRole, ActusDateTime
from jactus.observers import ConstantRiskFactorObserver


def create_loan_portfolio():
    """Create a portfolio of 5 PAM loans with varying terms."""
    loans = [
        {
            "contract_id": "LOAN-001",
            "notional": 100_000.0,
            "rate": 0.045,
            "maturity": ActusDateTime(2025, 1, 15),
        },
        {
            "contract_id": "LOAN-002",
            "notional": 250_000.0,
            "rate": 0.050,
            "maturity": ActusDateTime(2026, 6, 15),
        },
        {
            "contract_id": "LOAN-003",
            "notional": 500_000.0,
            "rate": 0.055,
            "maturity": ActusDateTime(2027, 1, 15),
        },
        {
            "contract_id": "LOAN-004",
            "notional": 75_000.0,
            "rate": 0.040,
            "maturity": ActusDateTime(2025, 7, 15),
        },
        {
            "contract_id": "LOAN-005",
            "notional": 1_000_000.0,
            "rate": 0.060,
            "maturity": ActusDateTime(2029, 1, 15),
        },
    ]

    contracts = []
    for loan in loans:
        attrs = ContractAttributes(
            contract_id=loan["contract_id"],
            contract_type=ContractType.PAM,
            contract_role=ContractRole.RPA,
            status_date=ActusDateTime(2024, 1, 1),
            initial_exchange_date=ActusDateTime(2024, 1, 15),
            maturity_date=loan["maturity"],
            notional_principal=loan["notional"],
            nominal_interest_rate=loan["rate"],
            interest_payment_cycle="6M",
        )
        contracts.append(attrs)

    return contracts


def simulate_portfolio_sequential():
    """Simulate each contract individually and aggregate results."""
    contracts = create_loan_portfolio()
    rf = ConstantRiskFactorObserver(constant_value=0.0)

    print("=" * 80)
    print("  Loan Portfolio Summary")
    print("=" * 80)
    print(f"\n{'ID':<12} {'Notional':>12} {'Rate':>8} {'Inflows':>15} {'Outflows':>15} {'Net':>15}")
    print("-" * 80)

    portfolio_inflows = 0.0
    portfolio_outflows = 0.0

    for attrs in contracts:
        contract = create_contract(attrs, rf)
        result = contract.simulate()

        inflows = sum(e.payoff for e in result.events if e.payoff > 0)
        outflows = sum(e.payoff for e in result.events if e.payoff < 0)
        net = inflows + outflows

        portfolio_inflows += inflows
        portfolio_outflows += outflows

        print(
            f"{attrs.contract_id:<12} "
            f"{attrs.notional_principal:>12,.0f} "
            f"{attrs.nominal_interest_rate:>7.2%} "
            f"{inflows:>15,.2f} "
            f"{outflows:>15,.2f} "
            f"{net:>15,.2f}"
        )

    portfolio_net = portfolio_inflows + portfolio_outflows
    print("-" * 80)
    print(
        f"{'TOTAL':<12} "
        f"{'':>12} "
        f"{'':>8} "
        f"{portfolio_inflows:>15,.2f} "
        f"{portfolio_outflows:>15,.2f} "
        f"{portfolio_net:>15,.2f}"
    )


def simulate_portfolio_batch():
    """Simulate portfolio using the batch portfolio API.

    Uses JACTUS's simulate_portfolio() for efficient batch execution.
    """
    try:
        from jactus.contracts.portfolio import simulate_portfolio
    except ImportError:
        print("Portfolio API not available. Using sequential simulation.")
        simulate_portfolio_sequential()
        return

    contracts = create_loan_portfolio()
    rf = ConstantRiskFactorObserver(constant_value=0.0)

    # Build (attrs, observer) pairs
    contract_pairs = [(attrs, rf) for attrs in contracts]

    results = simulate_portfolio(contract_pairs)

    print("\n" + "=" * 80)
    print("  Batch Portfolio Simulation Results")
    print("=" * 80)
    print(f"\n{'ID':<12} {'Events':>8} {'Non-Zero':>10} {'Net Cash Flow':>15}")
    print("-" * 50)

    for attrs, result in zip(contracts, results):
        n_events = len(result.events)
        non_zero = sum(1 for e in result.events if e.payoff != 0)
        net = sum(e.payoff for e in result.events)
        print(f"{attrs.contract_id:<12} {n_events:>8} {non_zero:>10} {net:>15,.2f}")


def portfolio_dv01():
    """Compute portfolio-level DV01 via finite difference."""
    contracts = create_loan_portfolio()
    bump = 0.0001  # 1 basis point

    def portfolio_pv(rate_bump=0.0):
        rf = ConstantRiskFactorObserver(constant_value=0.0)
        total = 0.0
        for attrs in contracts:
            bumped_attrs = ContractAttributes(
                **{
                    **{k: v for k, v in attrs.__dict__.items() if not k.startswith("_")},
                    "nominal_interest_rate": attrs.nominal_interest_rate + rate_bump,
                }
            )
            result = create_contract(bumped_attrs, rf).simulate()
            total += sum(e.payoff for e in result.events)
        return total

    pv_base = portfolio_pv(0.0)
    pv_bump = portfolio_pv(bump)
    dv01 = (pv_bump - pv_base) / bump

    print("\n" + "=" * 80)
    print("  Portfolio DV01 Analysis")
    print("=" * 80)
    print(f"  Portfolio PV (base):   {pv_base:>15,.2f}")
    print(f"  Portfolio PV (bumped): {pv_bump:>15,.2f}")
    print(f"  Portfolio DV01:        {dv01:>15,.4f}")
    print(f"  Interpretation: PV changes by {dv01:,.4f} per 1bp rate move")


if __name__ == "__main__":
    simulate_portfolio_sequential()
    simulate_portfolio_batch()
    portfolio_dv01()
