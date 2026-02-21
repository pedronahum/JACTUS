.. _api-reference:

API Reference
=============

This section contains the complete API reference for JACTUS.

.. toctree::
   :maxdepth: 2

Core Modules
------------

Core Types
^^^^^^^^^^

Time and DateTime
~~~~~~~~~~~~~~~~~

.. automodule:: jactus.core.time
   :members:
   :undoc-members:
   :show-inheritance:

Contract Attributes
~~~~~~~~~~~~~~~~~~~

.. automodule:: jactus.core.attributes
   :members:
   :undoc-members:
   :show-inheritance:

Contract State
~~~~~~~~~~~~~~

.. automodule:: jactus.core.states
   :members:
   :undoc-members:
   :show-inheritance:

Contract Events
~~~~~~~~~~~~~~~

.. automodule:: jactus.core.events
   :members:
   :undoc-members:
   :show-inheritance:

Contract Types
~~~~~~~~~~~~~~

.. automodule:: jactus.core.types
   :members:
   :undoc-members:
   :show-inheritance:

Utilities
^^^^^^^^^

Schedule Generation
~~~~~~~~~~~~~~~~~~~

.. automodule:: jactus.utilities.schedules
   :members:
   :undoc-members:
   :show-inheritance:

Business Day Conventions
~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: jactus.utilities.conventions
   :members:
   :undoc-members:
   :show-inheritance:

Calendars
~~~~~~~~~

.. automodule:: jactus.utilities.calendars
   :members:
   :undoc-members:
   :show-inheritance:

Mathematical Functions
~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: jactus.utilities.math
   :members:
   :undoc-members:
   :show-inheritance:

Contract Implementations
------------------------

Base Contract
^^^^^^^^^^^^^

.. automodule:: jactus.contracts.base
   :members:
   :undoc-members:
   :show-inheritance:

Factory Function
^^^^^^^^^^^^^^^^

.. automodule:: jactus.contracts
   :members: create_contract
   :undoc-members:

Principal Contracts
^^^^^^^^^^^^^^^^^^^

PAM - Principal at Maturity
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: jactus.contracts.pam
   :members:
   :undoc-members:
   :show-inheritance:

LAM - Linear Amortizer
~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: jactus.contracts.lam
   :members:
   :undoc-members:
   :show-inheritance:

LAX - Exotic Linear Amortizer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: jactus.contracts.lax
   :members:
   :undoc-members:
   :show-inheritance:

NAM - Negative Amortizer
~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: jactus.contracts.nam
   :members:
   :undoc-members:
   :show-inheritance:

ANN - Annuity
~~~~~~~~~~~~~

.. automodule:: jactus.contracts.ann
   :members:
   :undoc-members:
   :show-inheritance:

CLM - Call Money
~~~~~~~~~~~~~~~~

.. automodule:: jactus.contracts.clm
   :members:
   :undoc-members:
   :show-inheritance:

UMP - Undefined Maturity Profile
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: jactus.contracts.ump
   :members:
   :undoc-members:
   :show-inheritance:

Non-Principal Contracts
^^^^^^^^^^^^^^^^^^^^^^^

CSH - Cash
~~~~~~~~~~

.. automodule:: jactus.contracts.csh
   :members:
   :undoc-members:
   :show-inheritance:

STK - Stock
~~~~~~~~~~~

.. automodule:: jactus.contracts.stk
   :members:
   :undoc-members:
   :show-inheritance:

COM - Commodity
~~~~~~~~~~~~~~~

.. automodule:: jactus.contracts.com
   :members:
   :undoc-members:
   :show-inheritance:

Derivative Contracts
^^^^^^^^^^^^^^^^^^^^

FXOUT - Foreign Exchange Outright
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: jactus.contracts.fxout
   :members:
   :undoc-members:
   :show-inheritance:

OPTNS - Options
~~~~~~~~~~~~~~~

.. automodule:: jactus.contracts.optns
   :members:
   :undoc-members:
   :show-inheritance:

FUTUR - Futures
~~~~~~~~~~~~~~~

.. automodule:: jactus.contracts.futur
   :members:
   :undoc-members:
   :show-inheritance:

SWPPV - Plain Vanilla Swap
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: jactus.contracts.swppv
   :members:
   :undoc-members:
   :show-inheritance:

SWAPS - Generic Swap
~~~~~~~~~~~~~~~~~~~~

.. automodule:: jactus.contracts.swaps
   :members:
   :undoc-members:
   :show-inheritance:

CAPFL - Cap/Floor
~~~~~~~~~~~~~~~~~

.. automodule:: jactus.contracts.capfl
   :members:
   :undoc-members:
   :show-inheritance:

CEG - Credit Enhancement Guarantee
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: jactus.contracts.ceg
   :members:
   :undoc-members:
   :show-inheritance:

CEC - Credit Enhancement Collateral
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: jactus.contracts.cec
   :members:
   :undoc-members:
   :show-inheritance:

Contract Functions
------------------

Payoff Functions
^^^^^^^^^^^^^^^^

.. automodule:: jactus.functions.payoff
   :members:
   :undoc-members:
   :show-inheritance:

State Transition Functions
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. automodule:: jactus.functions.state
   :members:
   :undoc-members:
   :show-inheritance:

Observers
---------

Risk Factor Observer
^^^^^^^^^^^^^^^^^^^^

.. automodule:: jactus.observers.risk_factor
   :members:
   :undoc-members:
   :show-inheritance:

Child Contract Observer
^^^^^^^^^^^^^^^^^^^^^^^^

.. automodule:: jactus.observers.child_contract
   :members:
   :undoc-members:
   :show-inheritance:

Simulation Engine
-----------------

.. automodule:: jactus.engine.simulator
   :members:
   :undoc-members:
   :show-inheritance:

Exceptions
----------

.. automodule:: jactus.exceptions
   :members:
   :undoc-members:
   :show-inheritance:

Logging
-------

.. automodule:: jactus.logging_config
   :members:
   :undoc-members:
   :show-inheritance:

Quick Reference
---------------

Contract Type Summary
^^^^^^^^^^^^^^^^^^^^^

**Principal Contracts**

* **PAM** - Principal at Maturity (interest-only loans, bonds)
* **LAM** - Linear Amortizer (fixed principal amortization)
* **LAX** - Exotic Linear Amortizer (variable amortization schedules)
* **NAM** - Negative Amortizer (negative amortization loans)
* **ANN** - Annuity (level payment amortization)
* **CLM** - Call Money (callable overnight lending)
* **UMP** - Undefined Maturity Profile (open-ended deposits, revolving credit)

**Non-Principal Contracts**

* **CSH** - Cash (money market accounts, escrow)
* **STK** - Stock (equity positions)
* **COM** - Commodity (physical commodities)

**Derivative Contracts**

* **FXOUT** - Foreign Exchange Outright (FX forwards, swaps)
* **OPTNS** - Options (calls, puts, European/American)
* **FUTUR** - Futures (standardized forward contracts)
* **SWPPV** - Plain Vanilla Swap (fixed vs floating interest rate swaps)
* **SWAPS** - Generic Swap (cross-currency swaps, multi-leg swaps)
* **CAPFL** - Cap/Floor (interest rate caps and floors)
* **CEG** - Credit Enhancement Guarantee (credit protection)
* **CEC** - Credit Enhancement Collateral (collateral management)

Key Classes and Functions
^^^^^^^^^^^^^^^^^^^^^^^^^^

**Creating Contracts**::

    from jactus.contracts import create_contract
    from jactus.core import ContractAttributes, ContractType
    from jactus.observers import ConstantRiskFactorObserver

    attrs = ContractAttributes(
        contract_type=ContractType.PAM,
        # ... other attributes
    )
    rf_observer = ConstantRiskFactorObserver(0.05)
    contract = create_contract(attrs, rf_observer)

**Running Simulations**::

    # Initialize state
    state = contract.initialize_state()

    # Generate event schedule
    schedule = contract.generate_event_schedule()

    # Run simulation
    result = contract.simulate(rf_observer)

**Working with Time**::

    from jactus.core import ActusDateTime

    dt = ActusDateTime(2024, 1, 1, 0, 0, 0)

**Schedule Generation**::

    from jactus.utilities.schedules import generate_schedule

    schedule = generate_schedule(
        start=start_date,
        end=end_date,
        cycle="3M"  # Quarterly
    )

See Also
--------

* :doc:`../quickstart` - Get started with JACTUS
* :doc:`../guides/index` - User guides and tutorials
* :doc:`../spec/actus_overview` - ACTUS specification overview
* `GitHub Repository <https://github.com/pedronahum/jactus>`_
