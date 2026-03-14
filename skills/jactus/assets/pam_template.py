"""PAM (Principal at Maturity) contract template.

Demonstrates a bullet loan from both lender and borrower perspectives,
plus JAX sensitivity analysis (DV01).
"""

from jactus.contracts import create_contract
from jactus.core import ContractAttributes, ContractType, ContractRole, ActusDateTime
from jactus.observers import ConstantRiskFactorObserver


def create_pam_loan():
    """Create a $100,000 bullet loan at 5% for 1 year."""
    return ContractAttributes(
        contract_id="LOAN-001",
        contract_type=ContractType.PAM,
        contract_role=ContractRole.RPA,  # Lender perspective
        status_date=ActusDateTime(2024, 1, 1),
        initial_exchange_date=ActusDateTime(2024, 1, 15),
        maturity_date=ActusDateTime(2025, 1, 15),
        notional_principal=100_000.0,
        nominal_interest_rate=0.05,
        interest_payment_cycle="6M",
    )


def simulate_and_print(attrs, label=""):
    """Simulate a contract and print non-zero cash flow events."""
    rf = ConstantRiskFactorObserver(constant_value=0.0)
    contract = create_contract(attrs, rf)
    result = contract.simulate()

    if label:
        print(f"\n{'='*60}")
        print(f"  {label}")
        print(f"{'='*60}")

    print(f"{'Date':<25} {'Event':<6} {'Payoff':>15}")
    print("-" * 50)

    total = 0.0
    for event in result.events:
        if event.payoff != 0:
            print(f"{str(event.event_time):<25} {event.event_type.name:<6} {event.payoff:>15,.2f}")
            total += event.payoff

    print("-" * 50)
    print(f"{'Net Cash Flow':<31} {total:>15,.2f}")
    return result


def lender_perspective():
    """Simulate from the lender's perspective (RPA = Real Position Asset)."""
    attrs = create_pam_loan()
    simulate_and_print(attrs, "Lender Perspective (RPA)")


def borrower_perspective():
    """Simulate from the borrower's perspective (RPL = Real Position Liability)."""
    attrs = create_pam_loan()
    attrs = ContractAttributes(
        **{**attrs.__dict__, "contract_role": ContractRole.RPL}
    )
    simulate_and_print(attrs, "Borrower Perspective (RPL)")


def compute_dv01():
    """Compute DV01 (interest rate sensitivity) via finite difference."""
    base_rate = 0.05
    bump = 0.0001  # 1 basis point

    def total_cashflow(rate):
        attrs = ContractAttributes(
            contract_id="LOAN-001",
            contract_type=ContractType.PAM,
            contract_role=ContractRole.RPA,
            status_date=ActusDateTime(2024, 1, 1),
            initial_exchange_date=ActusDateTime(2024, 1, 15),
            maturity_date=ActusDateTime(2025, 1, 15),
            notional_principal=100_000.0,
            nominal_interest_rate=rate,
        )
        rf = ConstantRiskFactorObserver(constant_value=0.0)
        result = create_contract(attrs, rf).simulate()
        return sum(e.payoff for e in result.events)

    pv_base = total_cashflow(base_rate)
    pv_bump = total_cashflow(base_rate + bump)
    dv01 = (pv_bump - pv_base) / bump

    print(f"\n{'='*60}")
    print("  DV01 Sensitivity Analysis")
    print(f"{'='*60}")
    print(f"Base rate:     {base_rate:.4%}")
    print(f"Bump size:     {bump:.4%} (1 bp)")
    print(f"PV (base):     {pv_base:>15,.2f}")
    print(f"PV (bumped):   {pv_bump:>15,.2f}")
    print(f"DV01:          {dv01:>15,.4f}")


if __name__ == "__main__":
    lender_perspective()
    borrower_perspective()
    compute_dv01()
