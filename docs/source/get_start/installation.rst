Installation
===============

There are two recommended approaches for setting up the environment and installing the ccf2d package.

Prerequisites
-------------

* Python 3.12 or higher
* Git (for cloning the repository)

First, clone the repository and navigate to the project directory:

.. code-block:: bash

    git clone https://github.com/ytsimon2004/ccf2d.git
    cd ccf2d


UV Environment (Recommended)
-------------------------------

UV is a fast Python package installer and resolver. If you don't have UV installed, install it first:

.. seealso::

    https://docs.astral.sh/uv/getting-started/installation/

Then set up the environment:

.. code-block:: bash

    # Create virtual environment
    uv venv

    # Activate environment
    source .venv/bin/activate         # Linux/macOS
    .venv\Scripts\activate           # Windows

    # Install package in development mode
    uv pip install -e .

Conda Environment
------------------

If you prefer using Conda for environment management:

.. code-block:: bash

    # Create conda environment with Python 3.12
    conda create -n ccf2d python=3.12 -y

    # Activate environment
    conda activate ccf2d

    # Install package in development mode
    pip install -e .

Verification
------------

After installation, verify that ccf2d is working correctly:

.. code-block:: bash

    # Check if the command is available
    ccf2d --help

    # Initialize atlas data (this will download ~4.5GB)
    ccf2d init

.. note::
    
    If you encounter issues with ``llvmlite`` or ``numba`` dependencies on Python 3.12, this is a known compatibility issue with some atlas visualization libraries. The core registration functionality will still work.