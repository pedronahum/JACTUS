JACTUS Documentation
=======================

Welcome to the JACTUS documentation!

JACTUS is a high-performance implementation of the ACTUS (Algorithmic Contract
Types Unified Standards) specification using JAX for automatic differentiation
and GPU acceleration.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation
   quickstart
   api/index
   guides/index
   spec/actus_overview
   ARCHITECTURE
   ARRAY_MODE
   PAM
   derivatives

Features
--------

* **18 Contract Types**: Principal, non-principal, and derivative instruments
* **High Performance**: JAX JIT compilation, GPU/TPU acceleration, array-mode portfolio API
* **Differentiable**: Automatic differentiation for risk analytics (DV01, duration, sensitivities)
* **CLI**: Full ``jactus`` command-line interface for simulation, risk analytics, and portfolio management
* **MCP Server**: AI assistant integration via Model Context Protocol
* **Type Safety**: Full type annotations and mypy support
* **Well Tested**: 1400+ tests with cross-validation against ACTUS reference

Quick Links
-----------

* :ref:`installation`
* :ref:`quickstart`
* :ref:`api-reference`
* :ref:`user-guides`
* `GitHub Repository <https://github.com/pedronahum/JACTUS>`_
* `Issue Tracker <https://github.com/pedronahum/JACTUS/issues>`_

Examples
--------

**Interactive Jupyter Notebooks** (``examples/notebooks/``):

* **Annuity Mortgage** - 30-year mortgage with amortization visualization
* **Options Contracts** - Call/Put options with payoff diagrams
* **Interest Rate Cap** - Rate protection scenarios and calculations
* **Stock & Commodity** - Position tracking and derivative underliers

**Python Scripts** (``examples/``):

* PAM, LAM contracts
* Interest rate swaps
* FX swaps
* Cross-currency basis swaps

Indices and Tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
