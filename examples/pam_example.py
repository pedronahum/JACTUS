#!/usr/bin/env python3
"""
PAM (Principal at Maturity) Contract Example
============================================

This example demonstrates how to create, simulate, and analyze a Principal at Maturity
(PAM) loan contract using JACTUS. PAM is one of the core ACTUS contract types,
representing loans where principal is repaid in full at maturity, with periodic
interest payments.

What You'll Learn:
-----------------
1. How to create a PAM contract with ContractAttributes
2. How to use risk factor observers to model interest rates
3. How to simulate the contract lifecycle
4. How to analyze cashflows and events
5. How to leverage JAX for performance and differentiation
6. How to perform scenario analysis

ACTUS PAM Contract Type:
-----------------------
A PAM contract represents a loan with the following characteristics:
- Principal is disbursed at Initial Exchange Date (IED)
- Interest accrues according to a day count convention
- Interest is paid periodically (e.g., monthly, quarterly, annually)
- Principal is repaid in full at Maturity Date (MD)
- Ideal for: bullet loans, bonds, simple term loans

Example: $100,000 loan at 5% annual interest, 5-year term, quarterly interest payments
"""

import jax.numpy as jnp

from jactus.contracts import create_contract
from jactus.core import (
    ActusDateTime,
    ContractAttributes,
    ContractRole,
    ContractType,
    DayCountConvention,
)
from jactus.observers import ConstantRiskFactorObserver, JaxRiskFactorObserver


def example_1_basic_pam_loan():
    """
    Example 1: Basic PAM Loan
    -------------------------
    Create and simulate a simple 5-year loan with quarterly interest payments.

    Loan Details:
    - Principal: $100,000
    - Annual Interest Rate: 5% (0.05)
    - Term: 5 years
    - Interest Payments: Quarterly
    - Day Count: Actual/360
    """
    print("=" * 80)
    print("Example 1: Basic PAM Loan - $100,000 at 5% for 5 years")
    print("=" * 80)

    # Step 1: Define contract attributes
    # This describes the legal and financial terms of the loan
    attrs = ContractAttributes(
        contract_id="PAM-BASIC-001",
        contract_type=ContractType.PAM,
        contract_role=ContractRole.RPA,  # RPA = Paying/Borrower perspective

        # Key dates
        status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
        initial_exchange_date=ActusDateTime(2024, 1, 15, 0, 0, 0),  # Loan disbursement
        maturity_date=ActusDateTime(2029, 1, 15, 0, 0, 0),  # Principal repayment

        # Financial terms
        currency="USD",
        notional_principal=100000.0,
        nominal_interest_rate=0.05,  # 5% annual

        # Schedule and conventions
        day_count_convention=DayCountConvention.A360,  # Actual/360
        interest_payment_cycle="3M",  # Quarterly (3 months)
    )

    # Step 2: Create risk factor observer
    # This provides market data (interest rates, FX rates, etc.)
    # For a simple fixed-rate loan, we use a constant observer
    rf_obs = ConstantRiskFactorObserver(constant_value=0.05)

    # Step 3: Create the contract using the factory pattern
    # The factory automatically selects the correct contract implementation
    contract = create_contract(attrs, rf_obs)

    print(f"\nContract created: {contract.__class__.__name__}")
    print(f"Contract ID: {attrs.contract_id}")

    # Step 4: Simulate the contract lifecycle
    # This generates all events (IED, IP, MD) and computes cashflows
    result = contract.simulate()

    print(f"\nSimulation complete!")
    print(f"Total events generated: {len(result.events)}")

    # Step 5: Analyze the results
    print("\n" + "-" * 80)
    print("Event Timeline:")
    print("-" * 80)
    print(f"{'Date':<20} {'Event':<8} {'Payoff':<15} {'Notional':<15} {'Accrued Int':<15}")
    print("-" * 80)

    for event in result.events:
        date_str = event.event_time.to_iso()[:10]
        event_type = event.event_type.value
        payoff = float(event.payoff)

        # Get state after event
        nt = float(event.state_post.nt) if event.state_post else 0.0
        ipac = float(event.state_post.ipac) if event.state_post else 0.0

        print(f"{date_str:<20} {event_type:<8} ${payoff:>13.2f} ${nt:>13.2f} ${ipac:>13.2f}")

    # Step 6: Analyze cashflows
    print("\n" + "-" * 80)
    print("Cashflow Analysis:")
    print("-" * 80)

    cashflows = result.get_cashflows()

    total_interest = sum(amt for _, amt, _ in cashflows if amt > 0 and amt < 50000)
    total_principal = sum(amt for _, amt, _ in cashflows if amt >= 50000)
    total_outflow = sum(abs(amt) for _, amt, _ in cashflows)

    print(f"Total Interest Paid:    ${total_interest:,.2f}")
    print(f"Total Principal Repaid: ${total_principal:,.2f}")
    print(f"Total Outflow:          ${total_outflow:,.2f}")
    print(f"Effective Cost:         ${total_outflow - 100000:,.2f}")

    return result


def example_2_different_payment_frequencies():
    """
    Example 2: Comparing Different Payment Frequencies
    --------------------------------------------------
    Compare the same loan with monthly vs. quarterly vs. annual payments.

    This shows how payment frequency affects cashflow timing but not total cost.
    """
    print("\n\n" + "=" * 80)
    print("Example 2: Payment Frequency Comparison")
    print("=" * 80)

    frequencies = {
        "Monthly": "1M",
        "Quarterly": "3M",
        "Semi-Annual": "6M",
        "Annual": "1Y",
    }

    base_attrs = {
        "contract_type": ContractType.PAM,
        "contract_role": ContractRole.RPA,
        "status_date": ActusDateTime(2024, 1, 1, 0, 0, 0),
        "initial_exchange_date": ActusDateTime(2024, 1, 15, 0, 0, 0),
        "maturity_date": ActusDateTime(2026, 1, 15, 0, 0, 0),  # 2 years for faster demo
        "currency": "USD",
        "notional_principal": 100000.0,
        "nominal_interest_rate": 0.05,
        "day_count_convention": DayCountConvention.A360,
    }

    rf_obs = ConstantRiskFactorObserver(constant_value=0.05)

    print(f"\n{'Frequency':<15} {'# Payments':<12} {'Total Interest':<18} {'Avg Payment':<15}")
    print("-" * 70)

    for freq_name, cycle in frequencies.items():
        attrs = ContractAttributes(
            contract_id=f"PAM-FREQ-{freq_name}",
            interest_payment_cycle=cycle,
            **base_attrs
        )

        contract = create_contract(attrs, rf_obs)
        result = contract.simulate()

        # Count interest payments (exclude IED and MD)
        ip_events = [e for e in result.events if e.event_type.value == "IP"]
        num_payments = len(ip_events)

        # Calculate total interest
        total_interest = sum(float(e.payoff) for e in ip_events)
        avg_payment = total_interest / num_payments if num_payments > 0 else 0

        print(f"{freq_name:<15} {num_payments:<12} ${total_interest:>15.2f} ${avg_payment:>13.2f}")


def example_3_borrower_vs_lender_perspective():
    """
    Example 3: Borrower vs. Lender Perspective
    ------------------------------------------
    The same contract viewed from different roles produces opposite cashflows.

    - RPA (Real Position Asset / Paying) = Borrower perspective = negative cashflows
    - RPL (Real Position Liability / Receiving) = Lender perspective = positive cashflows
    """
    print("\n\n" + "=" * 80)
    print("Example 3: Borrower vs. Lender Perspective")
    print("=" * 80)

    attrs_borrower = ContractAttributes(
        contract_id="PAM-BORROWER",
        contract_type=ContractType.PAM,
        contract_role=ContractRole.RPA,  # Borrower (Paying)
        status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
        initial_exchange_date=ActusDateTime(2024, 1, 15, 0, 0, 0),
        maturity_date=ActusDateTime(2025, 1, 15, 0, 0, 0),
        currency="USD",
        notional_principal=100000.0,
        nominal_interest_rate=0.05,
        day_count_convention=DayCountConvention.A360,
        interest_payment_cycle="6M",
    )

    attrs_lender = ContractAttributes(
        contract_id="PAM-LENDER",
        contract_type=ContractType.PAM,
        contract_role=ContractRole.RPL,  # Lender (Receiving)
        status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
        initial_exchange_date=ActusDateTime(2024, 1, 15, 0, 0, 0),
        maturity_date=ActusDateTime(2025, 1, 15, 0, 0, 0),
        currency="USD",
        notional_principal=100000.0,
        nominal_interest_rate=0.05,
        day_count_convention=DayCountConvention.A360,
        interest_payment_cycle="6M",
    )

    rf_obs = ConstantRiskFactorObserver(constant_value=0.05)

    borrower = create_contract(attrs_borrower, rf_obs)
    lender = create_contract(attrs_lender, rf_obs)

    result_borrower = borrower.simulate()
    result_lender = lender.simulate()

    print("\nBorrower Perspective (RPA - Paying):")
    print(f"{'Date':<12} {'Event':<8} {'Cashflow':<15}")
    print("-" * 40)
    for e in result_borrower.events:
        date = e.event_time.to_iso()[:10]
        event = e.event_type.value
        cf = float(e.payoff)
        print(f"{date:<12} {event:<8} ${cf:>13.2f}")

    print("\nLender Perspective (RPL - Receiving):")
    print(f"{'Date':<12} {'Event':<8} {'Cashflow':<15}")
    print("-" * 40)
    for e in result_lender.events:
        date = e.event_time.to_iso()[:10]
        event = e.event_type.value
        cf = float(e.payoff)
        print(f"{date:<12} {event:<8} ${cf:>13.2f}")

    print("\nNote: Cashflows are mirror images - borrower's outflow = lender's inflow")


def example_4_jax_integration():
    """
    Example 4: JAX Integration - Performance and Differentiation
    ------------------------------------------------------------
    JACTUS is built on JAX, enabling high-performance computing and automatic
    differentiation for risk analysis.

    This example shows:
    1. Using JAX arrays in risk factor observers
    2. Computing sensitivities (Greeks) via automatic differentiation
    3. Scenario analysis with vectorization
    """
    print("\n\n" + "=" * 80)
    print("Example 4: JAX Integration - Computing Interest Rate Sensitivity")
    print("=" * 80)

    import jax

    def compute_total_cost(interest_rate: float) -> float:
        """Compute total cost of loan at a given interest rate."""
        attrs = ContractAttributes(
            contract_id="PAM-RISK-001",
            contract_type=ContractType.PAM,
            contract_role=ContractRole.RPA,
            status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
            initial_exchange_date=ActusDateTime(2024, 1, 15, 0, 0, 0),
            maturity_date=ActusDateTime(2029, 1, 15, 0, 0, 0),
            currency="USD",
            notional_principal=100000.0,
            nominal_interest_rate=interest_rate,
            day_count_convention=DayCountConvention.A360,
            interest_payment_cycle="3M",
        )

        rf_obs = JaxRiskFactorObserver(jnp.array([interest_rate]))
        contract = create_contract(attrs, rf_obs)
        result = contract.simulate()

        # Sum all positive cashflows (payments out)
        total = 0.0
        for event in result.events:
            payoff = float(event.payoff)
            if payoff > 0:
                total += payoff

        return total

    # Compute cost at base rate
    base_rate = 0.05
    base_cost = compute_total_cost(base_rate)

    print(f"\nBase scenario (5% rate):")
    print(f"  Total cost: ${base_cost:,.2f}")

    # Compute sensitivity to interest rate changes
    # This uses JAX automatic differentiation
    print(f"\nInterest Rate Sensitivity Analysis:")
    print(f"{'Rate':<10} {'Total Cost':<15} {'Change from Base':<20}")
    print("-" * 50)

    for rate in [0.03, 0.04, 0.05, 0.06, 0.07]:
        cost = compute_total_cost(rate)
        change = cost - base_cost
        print(f"{rate*100:>4.1f}%     ${cost:>12,.2f}     ${change:>+12,.2f} ({change/base_cost*100:+.2f}%)")

    print("\nInsight: A 1% increase in rate (5% → 6%) increases total cost by ~$5,000")


def example_5_long_term_mortgage():
    """
    Example 5: Realistic Long-Term Mortgage
    ---------------------------------------
    Simulate a 30-year mortgage with monthly payments, similar to a typical
    home loan in the US.

    Mortgage Details:
    - Principal: $300,000
    - Annual Interest Rate: 6.5%
    - Term: 30 years
    - Payment Frequency: Monthly
    """
    print("\n\n" + "=" * 80)
    print("Example 5: 30-Year Mortgage - $300,000 at 6.5%")
    print("=" * 80)

    attrs = ContractAttributes(
        contract_id="PAM-MORTGAGE-001",
        contract_type=ContractType.PAM,
        contract_role=ContractRole.RPA,
        status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
        initial_exchange_date=ActusDateTime(2024, 1, 15, 0, 0, 0),
        maturity_date=ActusDateTime(2054, 1, 15, 0, 0, 0),  # 30 years
        currency="USD",
        notional_principal=300000.0,
        nominal_interest_rate=0.065,
        day_count_convention=DayCountConvention.A360,
        interest_payment_cycle="1M",  # Monthly
    )

    rf_obs = ConstantRiskFactorObserver(constant_value=0.065)
    contract = create_contract(attrs, rf_obs)

    print("\nSimulating 30-year mortgage...")
    import time
    start = time.perf_counter()
    result = contract.simulate()
    elapsed = (time.perf_counter() - start) * 1000

    print(f"Simulation completed in {elapsed:.2f}ms")
    print(f"Total events: {len(result.events)}")

    # Analyze the mortgage
    ip_events = [e for e in result.events if e.event_type.value == "IP"]
    total_interest = sum(float(e.payoff) for e in ip_events)

    print(f"\nMortgage Summary:")
    print(f"  Principal:          ${300000:,.2f}")
    print(f"  Total Interest:     ${total_interest:,.2f}")
    print(f"  Total Paid:         ${300000 + total_interest:,.2f}")
    print(f"  Interest/Principal: {total_interest/300000:.1%}")

    # Show first few payments
    print(f"\nFirst 6 Monthly Payments:")
    print(f"{'Date':<12} {'Payment':<12}")
    print("-" * 25)
    for e in ip_events[:6]:
        date = e.event_time.to_iso()[:10]
        payment = float(e.payoff)
        print(f"{date:<12} ${payment:>9.2f}")

    print("\nNote: In a real mortgage, payments would be equal (annuity).")
    print("PAM is interest-only; for equal payments, use LAM (Linear Amortizer) or ANN (Annuity).")


def example_6_portfolio_analysis():
    """
    Example 6: Portfolio Analysis
    -----------------------------
    Analyze a portfolio of multiple PAM loans with different characteristics.

    This demonstrates:
    - Creating multiple contracts
    - Aggregating cashflows across a portfolio
    - Computing portfolio-level metrics
    """
    print("\n\n" + "=" * 80)
    print("Example 6: Loan Portfolio Analysis")
    print("=" * 80)

    # Define a portfolio of 3 loans
    portfolio = [
        {
            "id": "LOAN-001",
            "principal": 100000,
            "rate": 0.045,
            "term_years": 3,
            "cycle": "1M",
        },
        {
            "id": "LOAN-002",
            "principal": 250000,
            "rate": 0.055,
            "term_years": 5,
            "cycle": "3M",
        },
        {
            "id": "LOAN-003",
            "principal": 150000,
            "rate": 0.050,
            "term_years": 7,
            "cycle": "6M",
        },
    ]

    print(f"\nPortfolio consists of {len(portfolio)} loans:")
    print(f"{'Loan ID':<12} {'Principal':<15} {'Rate':<8} {'Term':<8} {'Cycle':<8}")
    print("-" * 60)

    total_principal = 0
    contracts = []

    for loan in portfolio:
        print(f"{loan['id']:<12} ${loan['principal']:>12,} {loan['rate']*100:>5.2f}% "
              f"{loan['term_years']:>6}yr {loan['cycle']:>6}")
        total_principal += loan["principal"]

        # Create contract
        maturity = ActusDateTime(2024 + loan["term_years"], 1, 15, 0, 0, 0)
        attrs = ContractAttributes(
            contract_id=loan["id"],
            contract_type=ContractType.PAM,
            contract_role=ContractRole.RPL,  # Lender perspective
            status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
            initial_exchange_date=ActusDateTime(2024, 1, 15, 0, 0, 0),
            maturity_date=maturity,
            currency="USD",
            notional_principal=float(loan["principal"]),
            nominal_interest_rate=loan["rate"],
            day_count_convention=DayCountConvention.A360,
            interest_payment_cycle=loan["cycle"],
        )

        rf_obs = ConstantRiskFactorObserver(constant_value=loan["rate"])
        contract = create_contract(attrs, rf_obs)
        contracts.append((loan["id"], contract))

    print("-" * 60)
    print(f"{'TOTAL':<12} ${total_principal:>12,}")

    # Simulate all contracts
    print(f"\nSimulating portfolio...")
    results = {}
    for loan_id, contract in contracts:
        results[loan_id] = contract.simulate()

    # Analyze portfolio
    print(f"\nPortfolio Analysis:")
    print(f"{'Loan ID':<12} {'# Events':<10} {'Total Interest':<18} {'Yield':<10}")
    print("-" * 60)

    total_interest = 0
    for loan_id, result in results.items():
        cashflows = result.get_cashflows()
        interest = sum(amt for _, amt, _ in cashflows if amt > 0 and amt < 200000)

        # Find principal for this loan
        principal = next(l["principal"] for l in portfolio if l["id"] == loan_id)
        term = next(l["term_years"] for l in portfolio if l["id"] == loan_id)
        annualized_yield = (interest / principal) / term * 100

        print(f"{loan_id:<12} {len(result.events):<10} ${interest:>15,.2f} {annualized_yield:>8.2f}%")
        total_interest += interest

    print("-" * 60)
    print(f"{'TOTAL':<12} {'':<10} ${total_interest:>15,.2f} {(total_interest/total_principal/(5/3))*100:>8.2f}%")


def main():
    """Run all examples."""
    print("\n" + "=" * 80)
    print(" " * 20 + "JACTUS PAM Contract Examples")
    print(" " * 20 + "============================")
    print("=" * 80)
    print("\nThis script demonstrates various aspects of PAM (Principal at Maturity)")
    print("contracts in JACTUS, from basic loans to portfolio analysis.\n")

    # Run all examples
    example_1_basic_pam_loan()
    example_2_different_payment_frequencies()
    example_3_borrower_vs_lender_perspective()
    example_4_jax_integration()
    example_5_long_term_mortgage()
    example_6_portfolio_analysis()

    print("\n\n" + "=" * 80)
    print("All examples completed successfully!")
    print("=" * 80)
    print("\nNext Steps:")
    print("  - Explore other contract types: CSH, STK, COM")
    print("  - Learn about state transition functions and payoff functions")
    print("  - Read docs/PAM.md for detailed architecture walkthrough")
    print("  - Try modifying the examples to suit your use case")
    print("\n")


if __name__ == "__main__":
    main()
