.. _installation:

Installation
============

Requirements
------------

* Python 3.10 or higher
* pip (Python package installer)

Basic Installation
------------------

Install jactus using pip:

.. code-block:: bash

    pip install jactus

Development Installation
------------------------

For development, clone the repository and install in editable mode:

.. code-block:: bash

    git clone https://github.com/pedronahum/JACTUS.git
    cd JACTUS
    pip install -e ".[dev,docs,viz]"

Optional Dependencies
---------------------

Visualization Tools
^^^^^^^^^^^^^^^^^^^

For plotting and visualization:

.. code-block:: bash

    pip install jactus[viz]

This installs pandas, matplotlib, and seaborn.

Documentation Tools
^^^^^^^^^^^^^^^^^^^

For building documentation:

.. code-block:: bash

    pip install jactus[docs]

All Dependencies
^^^^^^^^^^^^^^^^

To install all optional dependencies:

.. code-block:: bash

    pip install jactus[all]

GPU Support
-----------

For GPU acceleration, install JAX with CUDA support:

.. code-block:: bash

    # NVIDIA CUDA 13 (recommended)
    pip install jactus "jax[cuda13]"

    # NVIDIA CUDA 12
    pip install jactus "jax[cuda12]"

    # TPU
    pip install jactus "jax[tpu]"

**Float64 precision** (CPU/GPU only — TPUs do not support float64):

.. code-block:: python

    import jax
    jax.config.update("jax_enable_x64", True)
    import jactus  # import AFTER enabling

See the `JAX installation guide <https://jax.readthedocs.io/en/latest/installation.html>`_ for more details.

Verification
------------

Verify your installation:

.. code-block:: python

    import jactus
    print(jactus.__version__)
    # Output: 0.2.0

Test JAX is working:

.. code-block:: python

    import jax
    import jax.numpy as jnp

    x = jnp.array([1, 2, 3])
    print(jnp.sum(x))
    # Output: 6
