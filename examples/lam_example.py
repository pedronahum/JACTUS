#!/usr/bin/env python3
"""
LAM (Linear Amortizer) Contract Example
========================================

This example demonstrates how to create, simulate, and analyze a Linear Amortizer
(LAM) loan contract using JACTUS. LAM is one of the core ACTUS amortizing contract
types, representing loans where principal is repaid in fixed installments over time,
with interest calculated on a separate basis (IPCB).

What You'll Learn:
-----------------
1. How to create a LAM contract with fixed principal payments
2. How to use IPCB (Interest Calculation Base) modes: NT, NTIED, and NTL
3. How principal amortization reduces loan balance over time
4. How to analyze amortizing loan cashflows and schedules
5. How to compare different IPCB modes
6. How LAM differs from PAM (Principal at Maturity)

ACTUS LAM Contract Type:
------------------------
A LAM contract represents an amortizing loan with the following characteristics:
- Principal is disbursed at Initial Exchange Date (IED)
- Principal is repaid in fixed installments (PR events)
- Interest accrues and is paid based on IPCB (Interest Calculation Base)
- IPCB can track current notional (NT), stay fixed (NTIED), or lag (NTL)
- Notional decreases over time as principal is repaid
- Ideal for: auto loans, equipment financing, simple mortgages

Example: $120,000 loan at 6% annual interest, 5-year term, quarterly principal
         payments of $6,000 each
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


def example_1_basic_lam_loan():
    """
    Example 1: Basic LAM Loan with NT (Notional Tracking) Mode
    ----------------------------------------------------------
    Create and simulate a 5-year amortizing loan with quarterly principal payments.

    Loan Details:
    - Principal: $120,000
    - Annual Interest Rate: 6% (0.06)
    - Term: 5 years
    - Principal Payments: Quarterly ($6,000 each)
    - Interest Payments: Quarterly (on current notional)
    - IPCB Mode: NT (interest tracks current notional with 1-period lag)
    - Day Count: Actual/360
    """
    print("=" * 80)
    print("Example 1: Basic LAM Loan - $120,000 at 6% for 5 years")
    print("=" * 80)

    # Step 1: Define contract attributes
    attrs = ContractAttributes(
        contract_id="LAM-BASIC-001",
        contract_type=ContractType.LAM,
        contract_role=ContractRole.RPA,  # RPA = Paying/Borrower perspective
        # Key dates
        status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
        initial_exchange_date=ActusDateTime(2024, 1, 15, 0, 0, 0),  # Loan disbursement
        maturity_date=ActusDateTime(2029, 1, 15, 0, 0, 0),  # Final payment
        # Financial terms
        currency="USD",
        notional_principal=120000.0,
        nominal_interest_rate=0.06,  # 6% annual
        # Principal redemption schedule
        principal_redemption_cycle="3M",  # Quarterly (3 months)
        next_principal_redemption_amount=6000.0,  # Fixed $6,000 per quarter
        principal_redemption_anchor=ActusDateTime(2024, 4, 15, 0, 0, 0),
        # Interest calculation
        interest_payment_cycle="3M",  # Quarterly interest payments
        interest_calculation_base="NT",  # NT = Notional Tracking (default)
        day_count_convention=DayCountConvention.A360,
    )

    # Step 2: Create risk factor observer
    rf_obs = ConstantRiskFactorObserver(constant_value=0.06)

    # Step 3: Create the contract
    contract = create_contract(attrs, rf_obs)

    print(f"\nContract created: {contract.__class__.__name__}")
    print(f"Contract ID: {attrs.contract_id}")
    print(f"IPCB Mode: {attrs.interest_calculation_base}")

    # Step 4: Simulate the contract lifecycle
    result = contract.simulate()

    print("\nSimulation complete!")
    print(f"Total events generated: {len(result.events)}")

    # Step 5: Analyze the results
    print("\n" + "-" * 95)
    print("Event Timeline (First 8 events):")
    print("-" * 95)
    print(
        f"{'Date':<12} {'Event':<6} {'Payoff':<12} {'Notional':<12} "
        f"{'IPCB':<12} {'Accrued Int':<12}"
    )
    print("-" * 95)

    for event in result.events[:8]:
        date_str = event.event_time.to_iso()[:10]
        event_type = event.event_type.value
        payoff = float(event.payoff)

        # Get state after event
        if event.state_post:
            nt = float(event.state_post.nt)
            ipcb = float(event.state_post.ipcb) if hasattr(event.state_post, "ipcb") else nt
            ipac = float(event.state_post.ipac)
        else:
            nt, ipcb, ipac = 0.0, 0.0, 0.0

        print(
            f"{date_str:<12} {event_type:<6} ${payoff:>10.2f} "
            f"${nt:>10.2f} ${ipcb:>10.2f} ${ipac:>10.2f}"
        )

    # Step 6: Analyze cashflows
    print("\n" + "-" * 80)
    print("Cashflow Analysis:")
    print("-" * 80)

    # Get PR and IP events
    pr_events = [e for e in result.events if e.event_type.value == "PR"]
    ip_events = [e for e in result.events if e.event_type.value == "IP"]

    total_principal = sum(float(e.payoff) for e in pr_events)
    total_interest = sum(float(e.payoff) for e in ip_events)
    total_outflow = total_principal + total_interest

    print(f"Number of Principal Payments: {len(pr_events)}")
    print(f"Number of Interest Payments:  {len(ip_events)}")
    print("")
    print(f"Total Principal Repaid:       ${total_principal:,.2f}")
    print(f"Total Interest Paid:          ${total_interest:,.2f}")
    print(f"Total Outflow:                ${total_outflow:,.2f}")
    print(f"Effective Interest Cost:      {(total_interest/120000)*100:.2f}%")

    print("\nKey Insight: With NT mode, interest tracks the declining notional balance,")
    print("so interest payments decrease over time as principal is repaid.")

    return result


def example_2_ipcb_modes_comparison():
    """
    Example 2: Comparing IPCB Modes (NT, NTIED, NTL)
    ------------------------------------------------
    The Interest Calculation Base (IPCB) determines what amount is used for
    interest calculations. This dramatically affects cashflows!

    Three modes:
    - NT (Notional Tracking): Interest on notional from PREVIOUS period (1-period lag)
    - NTIED (Notional Tied): Interest on INITIAL notional (fixed, never changes)
    - NTL (Notional Lagged): Interest on notional at last IPCB event (custom lag)
    """
    print("\n\n" + "=" * 80)
    print("Example 2: IPCB Mode Comparison - NT vs NTIED vs NTL")
    print("=" * 80)

    base_attrs = {
        "contract_type": ContractType.LAM,
        "contract_role": ContractRole.RPA,
        "status_date": ActusDateTime(2024, 1, 1, 0, 0, 0),
        "initial_exchange_date": ActusDateTime(2024, 1, 15, 0, 0, 0),
        "maturity_date": ActusDateTime(2026, 1, 15, 0, 0, 0),  # 2 years
        "currency": "USD",
        "notional_principal": 100000.0,
        "nominal_interest_rate": 0.06,
        "principal_redemption_cycle": "3M",
        "next_principal_redemption_amount": 12500.0,  # $12.5k per quarter
        "principal_redemption_anchor": ActusDateTime(2024, 4, 15, 0, 0, 0),
        "interest_payment_cycle": "3M",
        "day_count_convention": DayCountConvention.A360,
    }

    rf_obs = ConstantRiskFactorObserver(constant_value=0.06)

    modes = {
        "NT": {
            "interest_calculation_base": "NT",
            "description": "Interest on notional from previous period (lagging)",
        },
        "NTIED": {
            "interest_calculation_base": "NTIED",
            "description": "Interest on initial notional (fixed at $100k)",
        },
        "NTL": {
            "interest_calculation_base": "NTL",
            "interest_calculation_base_cycle": "6M",  # IPCB fixed every 6 months
            "description": "Interest on notional at 6-month IPCB events",
        },
    }

    print("\n" + "-" * 80)
    print(f"{'Mode':<8} {'Description':<45} {'Total Interest':<15}")
    print("-" * 80)

    results_by_mode = {}

    for mode_name, mode_config in modes.items():
        attrs = ContractAttributes(contract_id=f"LAM-IPCB-{mode_name}", **base_attrs, **mode_config)

        contract = create_contract(attrs, rf_obs)
        result = contract.simulate()
        results_by_mode[mode_name] = result

        # Calculate total interest
        ip_events = [e for e in result.events if e.event_type.value == "IP"]
        total_interest = sum(float(e.payoff) for e in ip_events)

        print(f"{mode_name:<8} {mode_config['description']:<45} ${total_interest:>13,.2f}")

    # Detailed comparison
    print("\n" + "-" * 80)
    print("Detailed Cashflow Comparison (First 4 Quarters):")
    print("-" * 80)

    for mode_name in ["NT", "NTIED", "NTL"]:
        result = results_by_mode[mode_name]
        print(f"\n{mode_name} Mode:")
        print(f"{'Date':<12} {'Event':<6} {'Payoff':<12} {'Notional':<12} {'IPCB':<12}")
        print("-" * 60)

        for event in result.events[:9]:  # Show first 9 events
            date_str = event.event_time.to_iso()[:10]
            event_type = event.event_type.value
            payoff = float(event.payoff)

            if event.state_post:
                nt = float(event.state_post.nt)
                ipcb = float(event.state_post.ipcb) if hasattr(event.state_post, "ipcb") else nt
            else:
                nt, ipcb = 0.0, 0.0

            print(f"{date_str:<12} {event_type:<6} ${payoff:>10.2f} ${nt:>10.2f} ${ipcb:>10.2f}")

    print("\nKey Insights:")
    print("  - NT: Interest decreases as notional is paid down (most common)")
    print("  - NTIED: Interest stays constant (worst for borrower)")
    print("  - NTL: Interest updates at IPCB events only (hybrid approach)")


def example_3_auto_loan():
    """
    Example 3: Real-World Auto Loan
    -------------------------------
    Simulate a typical 5-year auto loan with monthly payments.

    Loan Details:
    - Car Price: $35,000
    - Down Payment: $5,000
    - Loan Amount: $30,000
    - Annual Interest Rate: 4.5%
    - Term: 5 years (60 months)
    - Monthly principal payment: $500
    """
    print("\n\n" + "=" * 80)
    print("Example 3: Auto Loan - $30,000 at 4.5% for 60 months")
    print("=" * 80)

    attrs = ContractAttributes(
        contract_id="LAM-AUTO-001",
        contract_type=ContractType.LAM,
        contract_role=ContractRole.RPA,
        status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
        initial_exchange_date=ActusDateTime(2024, 2, 1, 0, 0, 0),
        maturity_date=ActusDateTime(2029, 2, 1, 0, 0, 0),
        currency="USD",
        notional_principal=30000.0,
        nominal_interest_rate=0.045,
        principal_redemption_cycle="1M",  # Monthly
        next_principal_redemption_amount=500.0,  # $500/month
        principal_redemption_anchor=ActusDateTime(2024, 3, 1, 0, 0, 0),
        interest_payment_cycle="1M",
        interest_calculation_base="NT",
        day_count_convention=DayCountConvention.A360,
    )

    rf_obs = ConstantRiskFactorObserver(constant_value=0.045)
    contract = create_contract(attrs, rf_obs)

    print("\nSimulating 5-year auto loan...")
    import time

    start = time.perf_counter()
    result = contract.simulate()
    elapsed = (time.perf_counter() - start) * 1000

    print(f"Simulation completed in {elapsed:.2f}ms")
    print(f"Total events: {len(result.events)}")

    # Analyze the loan
    pr_events = [e for e in result.events if e.event_type.value == "PR"]
    ip_events = [e for e in result.events if e.event_type.value == "IP"]

    total_principal = sum(float(e.payoff) for e in pr_events)
    total_interest = sum(float(e.payoff) for e in ip_events)

    print("\nAuto Loan Summary:")
    print(f"  Loan Amount:        ${30000:,.2f}")
    print(f"  Number of Payments: {len(pr_events)}")
    print(f"  Total Interest:     ${total_interest:,.2f}")
    print(f"  Total Paid:         ${total_principal + total_interest:,.2f}")
    print(f"  Interest as % of Principal: {(total_interest/30000)*100:.2f}%")

    # Show payment schedule for first 6 months
    print("\nFirst 6 Monthly Payments:")
    print(
        f"{'Month':<8} {'Date':<12} {'Principal':<12} {'Interest':<12} {'Total':<12} {'Balance':<12}"
    )
    print("-" * 75)

    month = 1
    for i in range(len(result.events)):
        event = result.events[i]
        if event.event_type.value == "PR":
            pr_amount = float(event.payoff)
            date_str = event.event_time.to_iso()[:10]

            # Find corresponding IP event
            ip_amount = 0.0
            if i + 1 < len(result.events) and result.events[i + 1].event_type.value == "IP":
                ip_amount = float(result.events[i + 1].payoff)

            balance = float(event.state_post.nt) if event.state_post else 0.0
            total_payment = pr_amount + ip_amount

            print(
                f"{month:<8} {date_str:<12} ${pr_amount:>10.2f} ${ip_amount:>10.2f} "
                f"${total_payment:>10.2f} ${balance:>10.2f}"
            )

            month += 1
            if month > 6:
                break

    print("\nNote: Principal payment stays constant ($500/month),")
    print("      but interest decreases as balance reduces.")


def example_4_lam_vs_pam():
    """
    Example 4: LAM vs PAM Comparison
    --------------------------------
    Compare an amortizing loan (LAM) with an interest-only loan (PAM)
    to see the difference in cashflow patterns.
    """
    print("\n\n" + "=" * 80)
    print("Example 4: LAM vs PAM - Amortizing vs Interest-Only")
    print("=" * 80)

    base_config = {
        "status_date": ActusDateTime(2024, 1, 1, 0, 0, 0),
        "initial_exchange_date": ActusDateTime(2024, 1, 15, 0, 0, 0),
        "maturity_date": ActusDateTime(2026, 1, 15, 0, 0, 0),  # 2 years
        "currency": "USD",
        "notional_principal": 100000.0,
        "nominal_interest_rate": 0.06,
        "contract_role": ContractRole.RPA,
        "day_count_convention": DayCountConvention.A360,
    }

    rf_obs = ConstantRiskFactorObserver(constant_value=0.06)

    # Create LAM contract
    lam_attrs = ContractAttributes(
        contract_id="LAM-COMPARE-001",
        contract_type=ContractType.LAM,
        principal_redemption_cycle="6M",
        next_principal_redemption_amount=25000.0,  # $25k every 6 months
        principal_redemption_anchor=ActusDateTime(2024, 7, 15, 0, 0, 0),
        interest_payment_cycle="6M",
        interest_calculation_base="NT",
        **base_config,
    )

    # Create PAM contract
    pam_attrs = ContractAttributes(
        contract_id="PAM-COMPARE-001",
        contract_type=ContractType.PAM,
        interest_payment_cycle="6M",
        **base_config,
    )

    lam_contract = create_contract(lam_attrs, rf_obs)
    pam_contract = create_contract(pam_attrs, rf_obs)

    lam_result = lam_contract.simulate()
    pam_result = pam_contract.simulate()

    print("\nLAM (Amortizing Loan):")
    print(f"{'Date':<12} {'Event':<6} {'Payoff':<12} {'Notional':<12}")
    print("-" * 50)
    for e in lam_result.events:
        date = e.event_time.to_iso()[:10]
        event = e.event_type.value
        payoff = float(e.payoff)
        nt = float(e.state_post.nt) if e.state_post else 0.0
        print(f"{date:<12} {event:<6} ${payoff:>10.2f} ${nt:>10.2f}")

    print("\nPAM (Interest-Only Loan):")
    print(f"{'Date':<12} {'Event':<6} {'Payoff':<12} {'Notional':<12}")
    print("-" * 50)
    for e in pam_result.events:
        date = e.event_time.to_iso()[:10]
        event = e.event_type.value
        payoff = float(e.payoff)
        nt = float(e.state_post.nt) if e.state_post else 0.0
        print(f"{date:<12} {event:<6} ${payoff:>10.2f} ${nt:>10.2f}")

    # Compare totals
    lam_ip = sum(float(e.payoff) for e in lam_result.events if e.event_type.value == "IP")
    pam_ip = sum(float(e.payoff) for e in pam_result.events if e.event_type.value == "IP")

    print("\n" + "-" * 50)
    print("Comparison:")
    print("-" * 50)
    print(f"{'Metric':<30} {'LAM':<15} {'PAM':<15}")
    print("-" * 50)
    print(f"{'Principal Payments:':<30} {'Yes':<15} {'At maturity':<15}")
    print(f"{'Total Interest Paid:':<30} ${lam_ip:<13,.2f} ${pam_ip:<13,.2f}")
    print(f"{'Interest Savings with LAM:':<30} ${pam_ip - lam_ip:<13,.2f} {'':<15}")

    print("\nKey Insight: LAM saves interest by paying down principal early,")
    print("reducing the balance on which interest accrues.")


def example_5_equipment_financing():
    """
    Example 5: Equipment Financing with Large Final Balloon
    -------------------------------------------------------
    Simulate equipment financing where most principal is paid over time,
    but a large balloon payment remains at maturity.

    This shows flexibility of LAM for custom payment structures.
    """
    print("\n\n" + "=" * 80)
    print("Example 5: Equipment Financing with Balloon Payment")
    print("=" * 80)

    attrs = ContractAttributes(
        contract_id="LAM-EQUIPMENT-001",
        contract_type=ContractType.LAM,
        contract_role=ContractRole.RPA,
        status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
        initial_exchange_date=ActusDateTime(2024, 1, 15, 0, 0, 0),
        maturity_date=ActusDateTime(2027, 1, 15, 0, 0, 0),  # 3 years
        currency="USD",
        notional_principal=200000.0,  # $200k equipment
        nominal_interest_rate=0.055,  # 5.5%
        # Pay $5k per quarter for 3 years, leaves $140k balloon
        principal_redemption_cycle="3M",
        next_principal_redemption_amount=5000.0,
        principal_redemption_anchor=ActusDateTime(2024, 4, 15, 0, 0, 0),
        interest_payment_cycle="3M",
        interest_calculation_base="NT",
        day_count_convention=DayCountConvention.A360,
    )

    rf_obs = ConstantRiskFactorObserver(constant_value=0.055)
    contract = create_contract(attrs, rf_obs)
    result = contract.simulate()

    # Analyze payments
    pr_events = [e for e in result.events if e.event_type.value == "PR"]
    ip_events = [e for e in result.events if e.event_type.value == "IP"]

    regular_payments = sum(float(e.payoff) for e in pr_events[:-1])  # All but last
    balloon_payment = float(pr_events[-1].payoff) if pr_events else 0.0
    total_interest = sum(float(e.payoff) for e in ip_events)

    print("\nEquipment Financing Summary:")
    print(f"  Equipment Cost:         ${200000:,.2f}")
    print(f"  Regular Payments:       {len(pr_events) - 1} × $5,000")
    print(f"  Total Regular Principal: ${regular_payments:,.2f}")
    print(f"  Balloon Payment:        ${balloon_payment:,.2f}")
    print(f"  Total Interest:         ${total_interest:,.2f}")
    print(f"  Total Paid:             ${regular_payments + balloon_payment + total_interest:,.2f}")

    # Show payment breakdown
    print("\nPayment Structure:")
    print("  Quarterly payments (12 quarters): $5,000 principal + interest")
    print(f"  Final balloon payment: ${balloon_payment:,.2f}")
    print("")
    print("  This structure keeps quarterly payments low while paying off")
    print("  equipment over time. The balloon can be refinanced or paid from")
    print("  equipment sale/cash flow.")

    # Show last few payments
    print("\nLast 3 Payments:")
    print(f"{'Date':<12} {'Event':<6} {'Payoff':<12} {'Balance After':<15}")
    print("-" * 50)
    for e in result.events[-6:]:  # Last 6 events (3 PR + 3 IP/MD)
        date = e.event_time.to_iso()[:10]
        event = e.event_type.value
        payoff = float(e.payoff)
        balance = float(e.state_post.nt) if e.state_post else 0.0
        print(f"{date:<12} {event:<6} ${payoff:>10.2f} ${balance:>13.2f}")


def example_6_portfolio_amortizing_loans():
    """
    Example 6: Portfolio of Amortizing Loans
    ----------------------------------------
    Analyze a portfolio of LAM loans with different characteristics.
    """
    print("\n\n" + "=" * 80)
    print("Example 6: Portfolio of Amortizing Loans")
    print("=" * 80)

    portfolio = [
        {
            "id": "LAM-001",
            "principal": 50000,
            "rate": 0.045,
            "term_years": 3,
            "payment_amount": 4166.67,  # ~$4.2k per quarter
            "cycle": "3M",
        },
        {
            "id": "LAM-002",
            "principal": 100000,
            "rate": 0.055,
            "term_years": 5,
            "payment_amount": 5000.0,
            "cycle": "3M",
        },
        {
            "id": "LAM-003",
            "principal": 75000,
            "rate": 0.050,
            "term_years": 4,
            "payment_amount": 3125.0,
            "cycle": "6M",
        },
    ]

    print(f"\nPortfolio consists of {len(portfolio)} amortizing loans:")
    print(f"{'Loan ID':<10} {'Principal':<12} {'Rate':<8} {'Term':<8} {'Payment':<12} {'Cycle':<8}")
    print("-" * 70)

    total_principal = 0
    results = {}

    for loan in portfolio:
        print(
            f"{loan['id']:<10} ${loan['principal']:>9,} {loan['rate']*100:>5.2f}% "
            f"{loan['term_years']:>6}yr ${loan['payment_amount']:>9,.2f} {loan['cycle']:>6}"
        )
        total_principal += loan["principal"]

        # Create contract
        maturity = ActusDateTime(2024 + loan["term_years"], 1, 15, 0, 0, 0)
        attrs = ContractAttributes(
            contract_id=loan["id"],
            contract_type=ContractType.LAM,
            contract_role=ContractRole.RPL,  # Lender perspective
            status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
            initial_exchange_date=ActusDateTime(2024, 1, 15, 0, 0, 0),
            maturity_date=maturity,
            currency="USD",
            notional_principal=float(loan["principal"]),
            nominal_interest_rate=loan["rate"],
            principal_redemption_cycle=loan["cycle"],
            next_principal_redemption_amount=loan["payment_amount"],
            interest_payment_cycle=loan["cycle"],
            interest_calculation_base="NT",
            day_count_convention=DayCountConvention.A360,
        )

        rf_obs = ConstantRiskFactorObserver(constant_value=loan["rate"])
        contract = create_contract(attrs, rf_obs)
        results[loan["id"]] = contract.simulate()

    print("-" * 70)
    print(f"{'TOTAL':<10} ${total_principal:>9,}")

    # Analyze portfolio
    print("\nPortfolio Analysis:")
    print(
        f"{'Loan ID':<10} {'# Events':<10} {'Interest':<15} {'Principal':<15} {'Total Inflow':<15}"
    )
    print("-" * 75)

    total_interest = 0
    total_principal_received = 0

    for loan_id, result in results.items():
        ip_events = [e for e in result.events if e.event_type.value == "IP"]
        pr_events = [e for e in result.events if e.event_type.value == "PR"]

        interest = sum(float(e.payoff) for e in ip_events)
        principal = sum(float(e.payoff) for e in pr_events)

        print(
            f"{loan_id:<10} {len(result.events):<10} ${interest:>13,.2f} "
            f"${principal:>13,.2f} ${interest + principal:>13,.2f}"
        )

        total_interest += interest
        total_principal_received += principal

    print("-" * 75)
    print(
        f"{'TOTAL':<10} {'':<10} ${total_interest:>13,.2f} "
        f"${total_principal_received:>13,.2f} ${total_interest + total_principal_received:>13,.2f}"
    )

    print("\nPortfolio Metrics:")
    print(f"  Total Principal Outstanding: ${total_principal:,.2f}")
    print(f"  Expected Total Interest:     ${total_interest:,.2f}")
    print(f"  Portfolio Yield:             {(total_interest/total_principal)*100:.2f}%")


def main():
    """Run all examples."""
    print("\n" + "=" * 80)
    print(" " * 18 + "JACTUS LAM Contract Examples")
    print(" " * 18 + "============================")
    print("=" * 80)
    print("\nThis script demonstrates various aspects of LAM (Linear Amortizer)")
    print("contracts in JACTUS, including IPCB modes, auto loans, and portfolios.\n")

    # Run all examples
    example_1_basic_lam_loan()
    example_2_ipcb_modes_comparison()
    example_3_auto_loan()
    example_4_lam_vs_pam()
    example_5_equipment_financing()
    example_6_portfolio_amortizing_loans()

    print("\n\n" + "=" * 80)
    print("All examples completed successfully!")
    print("=" * 80)
    print("\nKey Takeaways:")
    print("  - LAM reduces principal over time via fixed payments")
    print("  - IPCB modes (NT/NTIED/NTL) control interest calculation")
    print("  - NT mode is most common: interest on previous period notional")
    print("  - LAM saves interest vs PAM by paying down principal early")
    print("  - LAM is ideal for auto loans, equipment financing, simple mortgages")
    print("\nNext Steps:")
    print("  - Explore ANN (Annuity) for constant total payments")
    print("  - Try NAM (Negative Amortizer) for payment < interest scenarios")
    print("  - See LAX for complex array-based schedules")
    print("  - Compare with examples/pam_example.py for interest-only loans")
    print("\n")


if __name__ == "__main__":
    main()
