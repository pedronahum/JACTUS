.. _quickstart:

Quick Start
===========

Get started with JACTUS in 5 minutes by creating your first financial contract.

Installation
------------

Install JACTUS using pip::

    pip install jactus

Or for development::

    git clone https://github.com/pedronahum/JACTUS.git
    cd JACTUS
    pip install -e ".[dev]"

Your First Contract
-------------------

Let's create a simple Principal at Maturity (PAM) contract - a 5-year interest-only loan.

Complete Example
^^^^^^^^^^^^^^^^

.. code-block:: python

    from jactus.contracts import create_contract
    from jactus.core import (
        ContractAttributes,
        ContractType,
        ContractRole,
        ActusDateTime,
        DayCountConvention,
        EndOfMonthConvention,
    )
    from jactus.observers import ConstantRiskFactorObserver

    # Step 1: Define contract attributes
    attrs = ContractAttributes(
        contract_id="LOAN-001",
        contract_type=ContractType.PAM,
        contract_role=ContractRole.RPA,  # Lender receives principal, pays interest
        status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
        initial_exchange_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
        maturity_date=ActusDateTime(2029, 1, 1, 0, 0, 0),  # 5 years
        notional_principal=100_000.0,  # $100,000 loan
        nominal_interest_rate=0.05,  # 5% annual interest
        day_count_convention=DayCountConvention.A360,  # Actual/360
        cycle_of_interest_payment="3M",  # Quarterly payments
        cycle_anchor_date_of_interest_payment=ActusDateTime(2024, 4, 1, 0, 0, 0),
        end_of_month_convention=EndOfMonthConvention.SD,  # Same day
        currency="USD",
    )

    # Step 2: Create risk factor observer (fixed interest rate)
    rf_observer = ConstantRiskFactorObserver(0.05)

    # Step 3: Create the contract
    contract = create_contract(attrs, rf_observer)

    # Step 4: Initialize state
    initial_state = contract.initialize_state()
    print(f"Initial notional: ${float(initial_state.nt):,.2f}")
    print(f"Interest rate: {float(initial_state.ipnr) * 100:.2f}%")

    # Step 5: Generate event schedule
    schedule = contract.generate_event_schedule()
    print(f"\nGenerated {len(schedule.events)} events")

    # Step 6: Display payment schedule
    print("\nPayment Schedule:")
    print("-" * 60)
    for event in schedule.events[:10]:  # Show first 10 events
        date = f"{event.event_time.year}-{event.event_time.month:02d}-{event.event_time.day:02d}"
        print(f"{event.event_type.value:<6} {date:<12} ${event.payoff:>12,.2f}")

    # Step 7: Run simulation
    result = contract.simulate(rf_observer)

    # Step 8: Analyze results
    total_interest = sum(
        event.payoff for event in schedule.events
        if event.event_type.value == "IP"
    )
    print(f"\nTotal interest over 5 years: ${total_interest:,.2f}")

Expected Output
^^^^^^^^^^^^^^^

Running this example produces::

    Initial notional: $100,000.00
    Interest rate: 5.00%

    Generated 22 events

    Payment Schedule:
    ------------------------------------------------------------
    IED    2024-01-01           $0.00
    IP     2024-04-01       $1,250.00
    IP     2024-07-01       $1,250.00
    IP     2024-10-01       $1,250.00
    IP     2025-01-01       $1,250.00
    IP     2025-04-01       $1,250.00
    IP     2025-07-01       $1,250.00
    IP     2025-10-01       $1,250.00
    IP     2026-01-01       $1,250.00
    IP     2026-04-01       $1,250.00

    Total interest over 5 years: $25,000.00

Understanding the Code
----------------------

Contract Attributes
^^^^^^^^^^^^^^^^^^^

The ``ContractAttributes`` class defines all contract parameters:

* **contract_type**: The ACTUS contract type (PAM = Principal at Maturity)
* **contract_role**: Perspective (RPA = lender, RPL = borrower)
* **notional_principal**: The loan amount ($100,000)
* **nominal_interest_rate**: Annual interest rate (5% = 0.05)
* **cycle_of_interest_payment**: Payment frequency ("3M" = quarterly)
* **maturity_date**: When the principal is repaid

Risk Factor Observer
^^^^^^^^^^^^^^^^^^^^

The ``ConstantRiskFactorObserver`` provides market data (interest rates, FX rates, etc.)::

    rf_observer = ConstantRiskFactorObserver(0.05)

For fixed-rate contracts, this remains constant. For floating-rate contracts, you would
implement a custom observer that returns different rates over time.

Event Schedule
^^^^^^^^^^^^^^

The event schedule contains all cash flow events:

* **IED**: Initial Exchange Date (loan disbursement)
* **IP**: Interest Payment (quarterly interest payments)
* **MD**: Maturity Date (principal repayment)

Each event has:

* ``event_type``: Type of event (IED, IP, MD, etc.)
* ``event_time``: When the event occurs
* ``payoff``: Cash flow amount (positive = receive, negative = pay)

Simulation
^^^^^^^^^^

The ``simulate()`` method processes all events and returns the final state::

    result = contract.simulate(rf_observer)

The simulation:

1. Starts with initial state
2. Processes each event in chronological order
3. Updates state variables at each event
4. Returns final contract state

Next Steps
----------

Explore More Examples
^^^^^^^^^^^^^^^^^^^^^

**Python Script Examples:**

All examples are in the ``examples/`` directory:

* ``pam_example.py`` - Principal at Maturity (bullet loans)
* ``lam_example.py`` - Linear Amortizer (equal principal payments)
* ``interest_rate_swap_example.py`` - Plain vanilla interest rate swap
* ``fx_swap_example.py`` - Foreign exchange swap
* ``cross_currency_basis_swap_example.py`` - Cross-currency basis swap

**Interactive Jupyter Notebooks:**

For hands-on learning, see the notebooks in ``examples/notebooks/``:

* ``01_annuity_mortgage.ipynb`` - ANN contract with 30-year mortgage visualization
* ``02_options_contracts.ipynb`` - OPTNS contracts with call/put payoff diagrams
* ``03_interest_rate_cap.ipynb`` - CAPFL contract with rate protection scenarios
* ``04_stock_commodity.ipynb`` - STK and COM contracts as derivative underliers

Try Different Contract Types
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Linear Amortizer (LAM)**::

    attrs = ContractAttributes(
        contract_type=ContractType.LAM,
        # ... same basic parameters ...
        cycle_of_principal_redemption="3M",
        next_principal_redemption_payment=5_000.0,  # $5k per quarter
    )

**Interest Rate Swap (SWPPV)**::

    attrs = ContractAttributes(
        contract_type=ContractType.SWPPV,
        # ... basic parameters ...
        nominal_interest_rate=0.05,  # Fixed leg
        nominal_interest_rate_2=0.03,  # Floating leg
        interest_payment_cycle="3M",
        rate_reset_cycle="3M",
    )

**FX Outright (FXOUT)**::

    attrs = ContractAttributes(
        contract_type=ContractType.FXOUT,
        currency="EUR",
        currency_2="USD",
        notional_principal=1_000_000.0,  # EUR 1M
        price_at_purchase_date=1.10,  # Spot rate
    )

Explore Time Features
^^^^^^^^^^^^^^^^^^^^^

JACTUS provides flexible time handling::

    from jactus.core import ActusDateTime

    # Create dates
    start = ActusDateTime(2024, 1, 1, 0, 0, 0)

    # Compare dates
    if start < end:
        print("Start before end")

    # Calculate differences
    from jactus.utilities.schedules import generate_schedule

    dates = generate_schedule(
        start=start,
        end=end,
        cycle="1M"  # Monthly
    )

Work with Schedules
^^^^^^^^^^^^^^^^^^^

Generate custom payment schedules::

    from jactus.utilities.schedules import generate_schedule

    schedule = generate_schedule(
        start=ActusDateTime(2024, 1, 1, 0, 0, 0),
        end=ActusDateTime(2029, 1, 1, 0, 0, 0),
        cycle="3M"  # Quarterly
    )

    for date in schedule:
        print(date)

**Cycle Notation:**

* ``1D`` = Daily
* ``1W`` = Weekly
* ``1M`` = Monthly
* ``3M`` = Quarterly
* ``6M`` = Semi-annual
* ``1Y`` = Annual

Common Patterns
---------------

Monthly Mortgage Payment
^^^^^^^^^^^^^^^^^^^^^^^^

Calculate monthly payments on a loan::

    attrs = ContractAttributes(
        contract_type=ContractType.PAM,
        notional_principal=300_000.0,  # $300k
        nominal_interest_rate=0.065,  # 6.5%
        cycle_of_interest_payment="1M",  # Monthly
        maturity_date=ActusDateTime(2054, 1, 1, 0, 0, 0),  # 30 years
    )

    contract = create_contract(attrs, ConstantRiskFactorObserver(0.065))
    schedule = contract.generate_event_schedule()

    # Find monthly payment
    ip_events = [e for e in schedule.events if e.event_type.value == "IP"]
    monthly_payment = ip_events[0].payoff
    print(f"Monthly payment: ${monthly_payment:,.2f}")

Analyzing Cash Flows
^^^^^^^^^^^^^^^^^^^^

Analyze different event types::

    schedule = contract.generate_event_schedule()

    # Group by event type
    events_by_type = {}
    for event in schedule.events:
        event_type = event.event_type.value
        if event_type not in events_by_type:
            events_by_type[event_type] = []
        events_by_type[event_type].append(event)

    # Calculate totals
    for event_type, events in events_by_type.items():
        total = sum(e.payoff for e in events)
        print(f"{event_type}: {len(events)} events, ${total:,.2f} total")

Comparing Scenarios
^^^^^^^^^^^^^^^^^^^

Compare different interest rates::

    rates = [0.03, 0.04, 0.05, 0.06, 0.07]

    for rate in rates:
        attrs.nominal_interest_rate = rate
        contract = create_contract(attrs, ConstantRiskFactorObserver(rate))
        schedule = contract.generate_event_schedule()

        total_interest = sum(
            e.payoff for e in schedule.events
            if e.event_type.value == "IP"
        )
        print(f"Rate {rate*100:.1f}%: Total interest = ${total_interest:,.2f}")

Troubleshooting
---------------

Common Issues
^^^^^^^^^^^^^

**"ValueError: maturity_date is required"**

Make sure all required attributes are provided::

    attrs = ContractAttributes(
        contract_type=ContractType.PAM,
        status_date=...,  # Required
        maturity_date=...,  # Required
        notional_principal=...,  # Required
        nominal_interest_rate=...,  # Required for PAM
    )

**"ValueError: Invalid cycle format"**

Use correct cycle notation::

    # Correct
    cycle_of_interest_payment="3M"  # Quarterly

    # Wrong
    cycle_of_interest_payment="P3M"  # Don't use 'P' prefix
    cycle_of_interest_payment="3m"  # Use uppercase 'M'

**"AttributeError: module has no attribute"**

Check imports::

    # Correct
    from jactus.core import ContractType

    # Wrong
    from jactus import ContractType  # Not exported at top level

More Examples
-------------

JACTUS includes comprehensive examples:

* ``examples/pam_example.py`` - PAM contracts (mortgages, bonds)
* ``examples/lam_example.py`` - LAM contracts (amortizing loans)
* ``examples/interest_rate_swap_example.py`` - Interest rate swaps
* ``examples/fx_swap_example.py`` - FX swaps
* ``examples/cross_currency_basis_swap_example.py`` - Cross-currency swaps

Run an example::

    python examples/pam_example.py

Further Reading
---------------

* :doc:`guides/index` - Comprehensive user guides
* :doc:`api/index` - Complete API reference
* :doc:`ARCHITECTURE` - System architecture
* :doc:`PAM` - Deep dive into PAM implementation
* :doc:`derivatives` - Derivative contracts guide

**External Resources:**

* `ACTUS Standard <https://www.actusfrf.org/>`_ - Official ACTUS specification
* `JAX Documentation <https://jax.readthedocs.io/>`_ - JAX framework documentation
* `GitHub Repository <https://github.com/pedronahum/jactus>`_ - Source code and issues

Need Help?
----------

* **Issues**: `GitHub Issues <https://github.com/pedronahum/jactus/issues>`_
* **Discussions**: `GitHub Discussions <https://github.com/pedronahum/jactus/discussions>`_
* **Email**: pnrodriguezh@gmail.com

You're now ready to start modeling financial contracts with JACTUS!
