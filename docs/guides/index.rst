.. _user-guides:

User Guides
===========

Comprehensive guides for using JACTUS to model financial contracts.

.. toctree::
   :maxdepth: 2

   getting_started
   principal_contracts
   derivative_contracts
   contract_composition
   schedule_generation
   advanced_topics

Getting Started
---------------

Introduction to JACTUS
^^^^^^^^^^^^^^^^^^^^^^

JACTUS is a JAX-based implementation of the ACTUS financial contract standard. It provides:

* **Type-safe contract modeling** with full validation
* **JAX integration** for automatic differentiation and GPU acceleration
* **18 contract types** covering principal, non-principal, and derivative instruments
* **Event-driven simulation** for cash flow generation
* **Composable contracts** for complex financial structures

Learning Resources
^^^^^^^^^^^^^^^^^^

**Interactive Notebooks**

For hands-on learning with visualizations, explore the Jupyter notebooks:

* ``examples/notebooks/01_annuity_mortgage.ipynb`` - 30-year mortgage with amortization charts
* ``examples/notebooks/02_options_contracts.ipynb`` - Call/Put options with payoff diagrams
* ``examples/notebooks/03_interest_rate_cap.ipynb`` - Interest rate protection scenarios
* ``examples/notebooks/04_stock_commodity.ipynb`` - Stock and commodity position tracking

**Python Examples**

Ready-to-run Python scripts are available in ``examples/``:

* ``pam_example.py`` - Principal at Maturity contracts
* ``lam_example.py`` - Linear Amortizer contracts
* ``interest_rate_swap_example.py`` - Plain vanilla swaps
* ``fx_swap_example.py`` - FX swaps
* ``cross_currency_basis_swap_example.py`` - Cross-currency swaps

Basic Workflow
^^^^^^^^^^^^^^

The typical workflow for using JACTUS:

1. **Define Contract Attributes**::

    from jactus.core import ContractAttributes, ContractType, ContractRole, ActusDateTime

    attrs = ContractAttributes(
        contract_id="LOAN-001",
        contract_type=ContractType.PAM,
        contract_role=ContractRole.RPA,  # Receive principal, pay interest
        status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
        # ... more attributes
    )

2. **Create Risk Factor Observer**::

    from jactus.observers import ConstantRiskFactorObserver

    rf_observer = ConstantRiskFactorObserver(0.05)  # 5% interest rate

3. **Create Contract**::

    from jactus.contracts import create_contract

    contract = create_contract(attrs, rf_observer)

4. **Generate Event Schedule**::

    schedule = contract.generate_event_schedule()
    for event in schedule.events:
        print(f"{event.event_type}: {event.event_time} -> ${event.payoff:.2f}")

5. **Run Simulation**::

    result = contract.simulate(rf_observer)

Principal Contracts Guide
-------------------------

PAM - Principal at Maturity
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Principal at Maturity contracts are interest-only loans where principal is repaid at maturity.

**Use Cases:**

* Interest-only mortgages
* Bullet bonds
* Short-term loans
* Treasury securities

**Example: 30-Year Mortgage**::

    from jactus.core import (
        ContractAttributes, ContractType, ContractRole, ActusDateTime,
        DayCountConvention, EndOfMonthConvention
    )
    from jactus.contracts import create_contract
    from jactus.observers import ConstantRiskFactorObserver

    # Define mortgage attributes
    attrs = ContractAttributes(
        contract_id="MORTGAGE-001",
        contract_type=ContractType.PAM,
        contract_role=ContractRole.RPA,  # Lender perspective
        status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
        initial_exchange_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
        maturity_date=ActusDateTime(2054, 1, 1, 0, 0, 0),  # 30 years
        notional_principal=500_000.0,  # $500k loan
        nominal_interest_rate=0.06,  # 6% annual
        day_count_convention=DayCountConvention.A360,
        cycle_of_interest_payment="1M",  # Monthly payments
        cycle_anchor_date_of_interest_payment=ActusDateTime(2024, 2, 1, 0, 0, 0),
        end_of_month_convention=EndOfMonthConvention.EOM,
        currency="USD",
    )

    # Create observer with fixed rate
    rf_observer = ConstantRiskFactorObserver(0.06)

    # Create contract and simulate
    contract = create_contract(attrs, rf_observer)
    schedule = contract.generate_event_schedule()
    result = contract.simulate(rf_observer)

    # Analyze results
    total_interest = sum(event.payoff for event in schedule.events
                        if event.event_type.value == "IP")
    print(f"Total interest over 30 years: ${total_interest:,.2f}")

**Key Parameters:**

* ``notional_principal``: Loan amount
* ``nominal_interest_rate``: Annual interest rate
* ``cycle_of_interest_payment``: Payment frequency (1M=monthly, 3M=quarterly, 1Y=annual)
* ``maturity_date``: When principal is repaid

**See:** ``examples/pam_example.py`` for complete examples

LAM - Linear Amortizer
^^^^^^^^^^^^^^^^^^^^^^^

Linear amortizer contracts have fixed principal payments plus decreasing interest.

**Use Cases:**

* Auto loans
* Equipment financing
* Fixed principal amortization schedules
* Construction loans

**Example: 5-Year Auto Loan**::

    attrs = ContractAttributes(
        contract_id="AUTO-001",
        contract_type=ContractType.LAM,
        contract_role=ContractRole.RPA,
        status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
        initial_exchange_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
        maturity_date=ActusDateTime(2029, 1, 1, 0, 0, 0),  # 5 years
        notional_principal=30_000.0,  # $30k car loan
        nominal_interest_rate=0.048,  # 4.8% APR
        cycle_of_interest_payment="1M",  # Monthly payments
        cycle_anchor_date_of_interest_payment=ActusDateTime(2024, 2, 1, 0, 0, 0),
        cycle_of_principal_redemption="1M",  # Monthly principal payments
        cycle_anchor_date_of_principal_redemption=ActusDateTime(2024, 2, 1, 0, 0, 0),
        next_principal_redemption_payment=500.0,  # $500/month principal
        currency="USD",
    )

**Amortization Modes:**

* ``NT`` (Notional): Principal amount per period
* ``NTIED`` (Notional IED): Based on initial notional
* ``NTL`` (Notional Last): Based on remaining notional

**See:** ``examples/lam_example.py`` for complete examples

Derivative Contracts Guide
---------------------------

Interest Rate Swaps (SWPPV)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Plain vanilla interest rate swaps exchange fixed and floating rate payments.

**Use Cases:**

* Convert fixed-rate debt to floating
* Convert floating-rate debt to fixed
* Interest rate risk management
* Speculation on rate movements

**Example: 5-Year Fixed vs Floating Swap**::

    attrs = ContractAttributes(
        contract_id="IRS-001",
        contract_type=ContractType.SWPPV,
        contract_role=ContractRole.RFL,  # Pay fixed, receive floating
        status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
        maturity_date=ActusDateTime(2029, 1, 1, 0, 0, 0),
        notional_principal=10_000_000.0,  # $10M notional
        nominal_interest_rate=0.05,  # 5% fixed rate
        nominal_interest_rate_2=0.03,  # 3% initial floating rate
        interest_payment_cycle="3M",  # Quarterly payments
        cycle_anchor_date_of_interest_payment=ActusDateTime(2024, 1, 1, 0, 0, 0),
        rate_reset_cycle="3M",  # Quarterly rate resets
        rate_reset_anchor=ActusDateTime(2024, 1, 1, 0, 0, 0),
        delivery_settlement="D",  # Net settlement
        currency="USD",
    )

**Settlement Options:**

* ``D`` (Net): Only pay/receive the difference
* ``S`` (Gross): Exchange both payments

**See:** ``examples/interest_rate_swap_example.py`` for detailed example

FX Derivatives (FXOUT)
^^^^^^^^^^^^^^^^^^^^^^^

Foreign exchange outright contracts for currency exchange at future dates.

**Use Cases:**

* FX hedging
* Forward FX contracts
* Currency swaps
* International payment hedging

**Example: EUR/USD FX Swap**::

    attrs = ContractAttributes(
        contract_id="FX-001",
        contract_type=ContractType.FXOUT,
        contract_role=ContractRole.RPA,
        status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
        maturity_date=ActusDateTime(2025, 1, 1, 0, 0, 0),
        settlement_date=ActusDateTime(2025, 1, 1, 0, 0, 0),
        currency="EUR",  # Primary currency
        currency_2="USD",  # Secondary currency
        notional_principal=1_000_000.0,  # EUR 1M
        notional_principal_2=1_100_000.0,  # USD 1.1M (at spot)
        delivery_settlement="S",  # Gross settlement
        purchase_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
        price_at_purchase_date=1.10,  # Spot rate
    )

    # Forward rate observer
    rf_observer = ConstantRiskFactorObserver(1.12)  # Forward rate

**See:** ``examples/fx_swap_example.py`` for complete example

Options (OPTNS)
^^^^^^^^^^^^^^^

Option contracts provide the right (not obligation) to buy/sell at a strike price.

**Types:**

* Call options (right to buy)
* Put options (right to sell)
* European (exercise at maturity only)
* American (exercise anytime)

**Example: European Call Option**::

    attrs = ContractAttributes(
        contract_id="OPT-001",
        contract_type=ContractType.OPTNS,
        contract_role=ContractRole.RPA,
        status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
        maturity_date=ActusDateTime(2025, 1, 1, 0, 0, 0),
        option_type="C",  # Call option
        option_strike_1=105_000.0,  # Strike price
        option_exercise_type="E",  # European
        settlement_currency="USD",
        currency="USD",
    )

**Option Parameters:**

* ``option_type``: "C" (call) or "P" (put)
* ``option_exercise_type``: "E" (European) or "A" (American)
* ``option_strike_1``: Primary strike price
* ``option_strike_2``: Secondary strike (for spreads)

Contract Composition Guide
---------------------------

Cross-Currency Basis Swaps (SWAPS)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Generic swaps support multi-leg, multi-currency structures.

**Use Cases:**

* Cross-currency basis swaps
* Multi-currency debt management
* Foreign subsidiary funding
* Basis trading

**Example: EUR vs USD Basis Swap**::

    from jactus.observers import MockChildContractObserver

    # Create child leg contracts (EUR and USD legs)
    eur_leg = create_contract(eur_leg_attrs, eur_rf_observer)
    usd_leg = create_contract(usd_leg_attrs, usd_rf_observer)

    # Register child contracts
    child_observer = MockChildContractObserver()
    child_observer.register_child(
        "EUR-LEG",
        events=list(eur_leg.generate_event_schedule().events),
        state=eur_leg.initialize_state(),
        attributes={"notional_principal": 10_000_000.0}
    )
    child_observer.register_child(
        "USD-LEG",
        events=list(usd_leg.generate_event_schedule().events),
        state=usd_leg.initialize_state(),
        attributes={"notional_principal": 11_000_000.0}
    )

    # Create parent swap contract
    swap_attrs = ContractAttributes(
        contract_id="XCCY-001",
        contract_type=ContractType.SWAPS,
        contract_role=ContractRole.RPA,
        status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
        maturity_date=ActusDateTime(2029, 1, 1, 0, 0, 0),
        contract_structure='{"FirstLeg": "EUR-LEG", "SecondLeg": "USD-LEG"}',
        delivery_settlement="D",
        currency="USD",
    )

    swap = create_contract(swap_attrs, rf_observer, child_observer)

**Composite Contract Types:**

* **SWAPS**: Multi-leg swaps
* **CAPFL**: Caps/floors on underlier
* **CEG**: Credit guarantees on covered contracts
* **CEC**: Collateral tracking for covered contracts

**See:** ``examples/cross_currency_basis_swap_example.py`` for detailed example

Credit Enhancement (CEG, CEC)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Credit enhancement contracts provide protection and collateral management.

**CEG - Credit Enhancement Guarantee:**

Provides credit protection on covered contracts::

    attrs = ContractAttributes(
        contract_id="CEG-001",
        contract_type=ContractType.CEG,
        contract_role=ContractRole.RPA,
        status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
        maturity_date=ActusDateTime(2029, 1, 1, 0, 0, 0),
        contract_structure='{"CoveredContract": "LOAN-001"}',
        coverage=0.8,  # 80% coverage
        credit_event_type="DL",  # Default
        credit_enhancement_guarantee_extent="NO",  # Notional only
        currency="USD",
    )

**CEC - Credit Enhancement Collateral:**

Tracks collateral against exposure::

    attrs = ContractAttributes(
        contract_id="CEC-001",
        contract_type=ContractType.CEC,
        contract_role=ContractRole.RPA,
        status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
        maturity_date=ActusDateTime(2029, 1, 1, 0, 0, 0),
        contract_structure='{"CoveredContract": "LOAN-001", "CoveringContract": "STK-001"}',
        coverage=1.2,  # 120% collateral requirement
        credit_enhancement_guarantee_extent="NO",
        currency="USD",
    )

Schedule Generation Guide
--------------------------

Understanding Cycles
^^^^^^^^^^^^^^^^^^^^

ACTUS uses cycle notation for recurring events:

* ``1D``: Daily
* ``1W``: Weekly
* ``1M``: Monthly
* ``3M``: Quarterly
* ``6M``: Semi-annual
* ``1Y``: Annual

**Stub Periods:**

* ``+``: Short stub at end
* ``-``: Short stub at beginning

**Example:**

* ``3M+``: Quarterly with short final period
* ``1M-``: Monthly with short initial period

Business Day Conventions
^^^^^^^^^^^^^^^^^^^^^^^^^

Handle non-business days:

* ``SCF``: Shift calendar following
* ``SCMF``: Shift calendar modified following
* ``CSF``: Calculate shift following
* ``CSMF``: Calculate shift modified following

**Example:**

If payment falls on Saturday, ``SCF`` moves to Monday.

Day Count Conventions
^^^^^^^^^^^^^^^^^^^^^

Calculate interest accrual:

* ``A360``: Actual/360
* ``A365``: Actual/365
* ``30E360``: 30E/360 (European)
* ``AA``: Actual/Actual

Advanced Topics
---------------

JAX Integration
^^^^^^^^^^^^^^^

All JACTUS contracts are JAX-compatible for automatic differentiation::

    import jax
    import jax.numpy as jnp

    def calculate_pv(interest_rate):
        attrs.nominal_interest_rate = float(interest_rate)
        contract = create_contract(attrs, ConstantRiskFactorObserver(float(interest_rate)))
        schedule = contract.generate_event_schedule()
        return sum(event.payoff for event in schedule.events)

    # Calculate sensitivity to interest rate
    interest_rate = 0.05
    pv_sensitivity = jax.grad(calculate_pv)(interest_rate)
    print(f"PV sensitivity to rates: {pv_sensitivity}")

Batch Processing
^^^^^^^^^^^^^^^^

Use JAX's vmap for vectorized operations::

    from jax import vmap

    rates = jnp.array([0.03, 0.04, 0.05, 0.06, 0.07])
    pvs = vmap(calculate_pv)(rates)

Performance Optimization
^^^^^^^^^^^^^^^^^^^^^^^^

Tips for optimal performance:

1. **Use JIT compilation** for repeated calculations
2. **Batch operations** with vmap when possible
3. **Pre-generate schedules** for multiple simulations
4. **Use constant observers** when rates don't change

Custom Risk Factor Observers
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Implement custom observers for dynamic rates::

    from jactus.observers.risk_factor import RiskFactorObserverProtocol

    class YieldCurveObserver:
        def __init__(self, curve):
            self.curve = curve

        def observe(self, market_object_code, time, states, attributes):
            # Interpolate rate from yield curve
            return self.curve.get_rate(time)

See Also
--------

* :doc:`../api/index` - Complete API reference
* :doc:`../derivatives` - Derivative contracts guide
* :doc:`../ARCHITECTURE` - System architecture
* :doc:`../PAM` - Deep dive into PAM implementation
* `Examples Directory <https://github.com/pedronahum/jactus/tree/main/examples>`_

Additional Resources
--------------------

**Working Examples:**

* ``examples/pam_example.py`` - PAM mortgages and bonds
* ``examples/lam_example.py`` - Amortizing loans
* ``examples/interest_rate_swap_example.py`` - Interest rate swaps
* ``examples/fx_swap_example.py`` - FX swaps
* ``examples/cross_currency_basis_swap_example.py`` - Cross-currency swaps

**Documentation:**

* ``docs/ARCHITECTURE.md`` - High-level architecture
* ``docs/PAM.md`` - PAM implementation walkthrough
* ``docs/derivatives.md`` - All derivative contract types

**External Resources:**

* `ACTUS Standard <https://www.actusfrf.org/>`_
* `JAX Documentation <https://jax.readthedocs.io/>`_
* `GitHub Repository <https://github.com/pedronahum/jactus>`_
