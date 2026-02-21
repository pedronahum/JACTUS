ACTUS Specification Overview
============================

The ACTUS Standard
------------------

ACTUS (Algorithmic Contract Types Unified Standards) is a standardized framework
for representing financial contracts as mathematical algorithms. It provides a unified
approach to modeling cash flows, risk analytics, and contract behavior across various
financial instruments.

**Key Benefits:**

* Consistent contract representation across institutions
* Standardized cash flow generation
* Risk analytics with common methodology
* Interoperability between systems
* Regulatory compliance support

JACTUS Implementation Status
-----------------------------

JACTUS implements 18 of the ACTUS contract types with full test coverage and
production-ready quality.

Implemented Contract Types
^^^^^^^^^^^^^^^^^^^^^^^^^^

**Principal Contracts (7/7)**

✅ **PAM - Principal at Maturity**
   * Status: Complete (100% coverage, 52 tests)
   * Use cases: Interest-only mortgages, bullet bonds, treasury securities
   * Features: Fixed or floating interest, flexible payment cycles
   * Example: ``examples/pam_example.py``

✅ **LAM - Linear Amortizer**
   * Status: Complete (95.65% coverage, 57 tests)
   * Use cases: Auto loans, equipment financing, construction loans
   * Features: Fixed principal payments, three amortization modes (NT, NTIED, NTL)
   * Example: ``examples/lam_example.py``

✅ **LAX - Exotic Linear Amortizer**
   * Status: Complete
   * Use cases: Exotic amortization schedules, stepped payments
   * Features: Array-based variable principal redemption schedules

✅ **NAM - Negative Amortizer**
   * Status: Complete
   * Use cases: Negative amortization loans, payment-capped mortgages
   * Features: Fixed payment amount with negative amortization support

✅ **ANN - Annuity**
   * Status: Complete
   * Use cases: Traditional mortgages, car loans, consumer loans
   * Features: Level payment amortization

✅ **CLM - Call Money**
   * Status: Complete
   * Use cases: Call money markets, overnight lending
   * Features: Undefined maturity, callable at any time

✅ **UMP - Undefined Maturity Profile**
   * Status: Complete
   * Use cases: Revolving credit, open-ended deposits
   * Features: No fixed maturity date, flexible cash flows

**Non-Principal Contracts (3/3)**

✅ **CSH - Cash**
   * Status: Complete (100% coverage, 17 tests)
   * Use cases: Money market accounts, escrow accounts, cash positions
   * Features: Simple cash flow tracking

✅ **STK - Stock**
   * Status: Complete (100% coverage, 18 tests)
   * Use cases: Equity positions, dividend tracking
   * Features: Dividend payments, stock splits, purchase/sale

✅ **COM - Commodity**
   * Status: Complete (100% coverage, 18 tests)
   * Use cases: Physical commodities, commodity positions
   * Features: Quantity-based tracking, price movements

**Derivative Contracts (8/8)**

✅ **FXOUT - Foreign Exchange Outright**
   * Status: Complete (100% coverage, 20 tests)
   * Use cases: FX forwards, currency swaps, FX hedging
   * Features: Dual currency, spot/forward rates, net/gross settlement
   * Example: ``examples/fx_swap_example.py``

✅ **OPTNS - Options**
   * Status: Complete (97.20% coverage, 25 tests)
   * Use cases: Call/put options, spreads, hedging strategies
   * Features: European/American exercise, multiple strikes

✅ **FUTUR - Futures**
   * Status: Complete (97.22% coverage, 20 tests)
   * Use cases: Commodity futures, index futures, hedging
   * Features: Mark-to-market, standardized contracts

✅ **SWPPV - Plain Vanilla Swap**
   * Status: Complete (95.12% coverage, 22 tests)
   * Use cases: Fixed vs floating interest rate swaps
   * Features: Net/gross settlement, flexible rate resets, both legs
   * Example: ``examples/interest_rate_swap_example.py``

✅ **SWAPS - Generic Swap**
   * Status: Complete (93.38% coverage, 22 tests)
   * Use cases: Cross-currency swaps, multi-leg swaps, basis swaps
   * Features: Multiple legs, contract composition, multi-currency
   * Example: ``examples/cross_currency_basis_swap_example.py``

✅ **CAPFL - Cap/Floor**
   * Status: Complete (96.08% coverage, 19 tests)
   * Use cases: Interest rate caps, floors, collars
   * Features: References underlier, cap/floor strikes

✅ **CEG - Credit Enhancement Guarantee**
   * Status: Complete (94.74% coverage, 30 tests)
   * Use cases: Credit protection, loan guarantees, portfolio protection
   * Features: Multiple covered contracts, coverage ratios, credit events

✅ **CEC - Credit Enhancement Collateral**
   * Status: Complete (96.00% coverage, 31 tests)
   * Use cases: Margin accounts, collateralized lending, repos
   * Features: Collateral tracking, sufficiency checking, multiple contracts

**Summary:**

* ✅ **Implemented**: 18 contract types (95%+ coverage)
* 📊 **Total Tests**: 1,192+ tests across all contract types
* 🎯 **Overall Coverage**: 95%+ on all implemented contracts

Core ACTUS Concepts
-------------------

Contract Attributes
^^^^^^^^^^^^^^^^^^^

ACTUS defines numerous contract attributes organized into categories (JACTUS implements ~80):

**Contract Identification**

* ``contract_id``: Unique contract identifier
* ``contract_type``: Type (PAM, LAM, SWPPV, etc.)
* ``contract_role``: Perspective (RPA = asset, RPL = liability)
* ``status_date``: Contract status/calculation date

**Calendar & Timing**

* ``initial_exchange_date``: When contract starts
* ``maturity_date``: When contract ends
* ``cycle_anchor_date_*``: Anchors for recurring events
* ``cycle_of_*``: Frequency specifications (1M, 3M, 1Y)

**Notional & Rates**

* ``notional_principal``: Contract amount
* ``nominal_interest_rate``: Interest rate (fixed or initial floating)
* ``day_count_convention``: Interest calculation method
* ``end_of_month_convention``: Month-end handling

**Settlement & Delivery**

* ``delivery_settlement``: Net (D) or gross (S) settlement
* ``settlement_currency``: Currency for settlements
* ``purchase_date``: Purchase/trade date
* ``price_at_purchase_date``: Purchase price

JACTUS supports all required ACTUS attributes with full validation.

State Variables
^^^^^^^^^^^^^^^

Contract state tracks the current condition of a contract:

**Core State Variables:**

* ``nt``: Notional principal (current amount)
* ``ipnr``: Interest payment nominal rate
* ``ipac``: Interest payment accrued
* ``feac``: Fee accrued
* ``nsc``: Notional scaling multiplier
* ``isc``: Interest scaling multiplier
* ``prf``: Performance (PF/DL/DQ/DF)
* ``sd``: Status date
* ``tmd``: Maturity date
* ``ipcl``: Interest payment calculation base

All state variables are JAX arrays for automatic differentiation.

Event Types
^^^^^^^^^^^

ACTUS defines 30+ event types for different contract actions:

**Common Event Types:**

* ``IED``: Initial Exchange Date (contract start)
* ``IP``: Interest Payment
* ``IPFX``: Interest Payment Fixed Leg
* ``IPFL``: Interest Payment Floating Leg
* ``PR``: Principal Redemption
* ``MD``: Maturity Date
* ``RR``: Rate Reset
* ``XD``: Exercise Date (options)
* ``STD``: Settlement (derivatives)
* ``DV``: Dividend Payment (stocks)

Events are generated according to cycles and conventions.

State Transition Functions
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Each contract type has state transition functions (STF) that compute
the contract's state after each event:

**Process:**

1. Current state + Event → State Transition Function → New state
2. States are immutable (functional programming)
3. All transitions are deterministic
4. JAX-compatible for differentiation

**Example:**

.. code-block:: python

    new_state = stf_pam_ip(
        event=interest_payment_event,
        state=current_state,
        attributes=contract_attributes,
        risk_factor=interest_rate
    )

JACTUS implements STFs for all contract types following ACTUS specifications.

Payoff Functions
^^^^^^^^^^^^^^^^

Payoff functions (POF) determine the cash flows generated by contract events:

**Calculation:**

* Cash flow = Payoff Function(Event, State, Attributes, Risk Factors)
* Positive = receive cash
* Negative = pay cash
* Currency-specific

**Example:**

.. code-block:: python

    payoff = pof_pam_ip(
        state=current_state,
        attributes=contract_attributes,
        risk_factor=interest_rate,
        time=event_time
    )

JACTUS implements POFs for all event types across all contracts.

Business Day Conventions
^^^^^^^^^^^^^^^^^^^^^^^^

Handle non-business days (weekends, holidays):

**Shift Conventions:**

* ``SCF``: Shift Calendar Following (move to next business day)
* ``SCMF``: Shift Calendar Modified Following (stay in same month)
* ``SCBW``: Shift Calendar Backward (move to previous business day)
* ``SCMBW``: Shift Calendar Modified Backward

**Calculate Conventions:**

* ``CSF``: Calculate Shift Following
* ``CSMF``: Calculate Shift Modified Following
* ``CSBW``: Calculate Shift Backward
* ``CSMBW``: Calculate Shift Modified Backward

JACTUS implements all ACTUS conventions.

Day Count Conventions
^^^^^^^^^^^^^^^^^^^^^

Calculate interest accrual between dates:

**Implemented Conventions:**

* ``A/360``: Actual/360 (money market)
* ``A/365``: Actual/365 (UK bonds)
* ``30/360``: 30/360 US (corporate bonds)
* ``30E/360``: 30E/360 European
* ``A/A``: Actual/Actual (government bonds)
* ``B/252``: Business/252 (Brazilian)

Each convention has specific calculation rules per ACTUS.

End-of-Month Conventions
^^^^^^^^^^^^^^^^^^^^^^^^

Handle month-end dates:

* ``SD``: Same Day (keep day of month)
* ``EOM``: End of Month (move to month end if applicable)

**Example:**

If cycle anchor is Jan 31 with EOM:
* Feb payment → Feb 28 (or 29 in leap year)
* Mar payment → Mar 31
* Apr payment → Apr 30

JACTUS JAX Integration
----------------------

Differentiable Contracts
^^^^^^^^^^^^^^^^^^^^^^^^

All JACTUS contracts support automatic differentiation:

.. code-block:: python

    import jax
    import jax.numpy as jnp

    def contract_pv(interest_rate):
        contract = create_pam_contract(rate=interest_rate)
        cashflows = contract.generate_cashflows()
        return jnp.sum(cashflows)

    # Calculate sensitivity
    rate = 0.05
    pv_sensitivity = jax.grad(contract_pv)(rate)

**Applications:**

* Duration and convexity calculation
* Greeks for derivatives
* Risk sensitivities
* Portfolio optimization

Vectorized Operations
^^^^^^^^^^^^^^^^^^^^^

Use JAX's vmap for batch processing:

.. code-block:: python

    from jax import vmap

    # Process multiple rates simultaneously
    rates = jnp.array([0.03, 0.04, 0.05, 0.06, 0.07])
    pvs = vmap(contract_pv)(rates)

**Benefits:**

* Parallel computation
* GPU acceleration
* Efficient batch processing

JIT Compilation
^^^^^^^^^^^^^^^

Just-in-time compilation for performance:

.. code-block:: python

    from jax import jit

    # Compile function
    fast_pv = jit(contract_pv)

    # First call compiles, subsequent calls are fast
    result = fast_pv(0.05)

**Performance:**

* 10-100x speedup typical
* GPU execution when available
* Minimal memory overhead

Implementation Notes
--------------------

ACTUS Compliance
^^^^^^^^^^^^^^^^

JACTUS follows ACTUS Technical Specification v1.1-843f7a3-2020-06-08:

* ✅ Contract attribute naming and structure
* ✅ State transition functions per specification
* ✅ Payoff function calculations
* ✅ Event scheduling algorithms
* ✅ Business day and day count conventions
* ✅ End-of-month handling

**Deviations:**

* Uses JAX arrays instead of scalar values (for differentiation)
* Functional programming approach (immutable states)
* Factory pattern for contract creation

Type Safety
^^^^^^^^^^^

JACTUS uses Pydantic for full type validation:

* All attributes validated at creation
* Type errors caught early
* IDE autocomplete support
* mypy static type checking

Testing Strategy
^^^^^^^^^^^^^^^^

Comprehensive testing approach:

* **Unit Tests**: Each contract type fully tested
* **Integration Tests**: Cross-contract composition validated
* **Property Tests**: Hypothesis-based testing for core functions
* **Performance Tests**: Benchmarking framework

References
----------

* `ACTUS Technical Specification v1.1 <https://www.actusfrf.org/>`_
* `ACTUS Data Dictionary <https://www.actusfrf.org/data-dictionary>`_
* `ACTUS Algorithmic Standard <https://www.actusfrf.org/algorithmic-standard>`_
* `JAX Documentation <https://jax.readthedocs.io/>`_

Further Reading
---------------

* :doc:`../guides/index` - User guides and tutorials
* :doc:`../api/index` - Complete API reference
* :doc:`../ARCHITECTURE` - System architecture details
* :doc:`../PAM` - PAM implementation deep dive
* :doc:`../derivatives` - Derivative contracts guide

See Also
--------

**JACTUS Documentation:**

* :doc:`../quickstart` - Get started in 5 minutes
* :doc:`../installation` - Installation guide
* `GitHub Repository <https://github.com/pedronahum/jactus>`_

**Working Examples:**

* ``examples/pam_example.py`` - Principal at Maturity
* ``examples/lam_example.py`` - Linear Amortizer
* ``examples/interest_rate_swap_example.py`` - Interest Rate Swaps
* ``examples/fx_swap_example.py`` - FX Swaps
* ``examples/cross_currency_basis_swap_example.py`` - Cross-Currency Swaps
