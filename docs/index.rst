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

Features
--------

* **High Performance**: Leverages JAX for JIT compilation and GPU acceleration
* **Type Safety**: Full type annotations and mypy support
* **Comprehensive**: Implements the complete ACTUS standard
* **Differentiable**: Automatic differentiation for risk analytics
* **Well Tested**: Extensive test coverage and validation

Quick Links
-----------

* :ref:`installation`
* :ref:`quickstart`
* :ref:`api-reference`
* :ref:`user-guides`
* `GitHub Repository <https://github.com/pedronahum/jactus>`_
* `Issue Tracker <https://github.com/pedronahum/jactus/issues>`_

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
