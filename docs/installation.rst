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

    # For CUDA 12
    pip install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

    # For CUDA 11
    pip install --upgrade "jax[cuda11_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

See the `JAX installation guide <https://github.com/google/jax#installation>`_ for more details.

Verification
------------

Verify your installation:

.. code-block:: python

    import jactus
    print(jactus.__version__)
    # Output: 0.1.2

Test JAX is working:

.. code-block:: python

    import jax
    import jax.numpy as jnp

    x = jnp.array([1, 2, 3])
    print(jnp.sum(x))
    # Output: 6
