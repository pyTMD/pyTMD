=======
Testing
=======

``pyTMD`` uses the ``pytest`` framework to run tests and verify outputs.
Running the test suite requires a `dev installation <../getting_started/Install.html>`_ of the ``pyTMD`` package to include all of the optional dependencies.

.. code-block:: bash

    python -m pip install --editable '.[dev]'

Running the Test Suite
^^^^^^^^^^^^^^^^^^^^^^

Using the ``pytest`` command:

.. code-block:: bash

    pytest --directory <path_to_tide_models> test/

Using ``pixi``:

.. code-block:: bash

    pixi run test "--directory <path_to_tide_models>"

The test suite is run in verbose mode as a default.

Coverage Reports
^^^^^^^^^^^^^^^^

Coverage reports can be generated using the ``pytest-cov`` plugin (which is installed with the dev installation).

.. code-block:: bash

    pytest --cov pyTMD --cov-report=term 

.. code-block:: bash

    pixi run test "--cov ../pyTMD --cov-report=term"

Parallelization
^^^^^^^^^^^^^^^

As a default, the ``pytest`` suite is run in parallel using the ``pytest-xdist`` plugin (which is also installed with the dev installation).
To run in series and disable parallelization, set the number of processes to 0:

.. code-block:: bash

    pytest --n 0

.. code-block:: bash

    pixi run test "-n 0"

Continuous Integration
^^^^^^^^^^^^^^^^^^^^^^
We use `GitHub Actions <https://github.com/pyTMD/pyTMD/actions>`_ continuous integration (CI) services to build and test the project on Linux (``ubuntu-latest``), Mac (``macos-latest``) and Windows (``windows-latest``) Operating Systems.
The configuration files for this service are in the `GitHub workflows <https://github.com/pyTMD/pyTMD/tree/main/.github/workflows>`_ directory.
The workflows use ``mamba`` and the `environment.yml <https://github.com/pyTMD/pyTMD/blob/main/environment.yml>`_ file to install the required dependencies and build the environment.

The GitHub Actions jobs include:

* Running `flake8 <https://flake8.pycqa.org/en/latest/>`_ to check the code for style and compilation errors
* Running the test suite on multiple combinations of OS and Python version
* Uploading test coverage statistics to `Codecov <https://app.codecov.io/gh/pyTMD/pyTMD>`_
* Uploading source and wheel distributions to `PyPI <https://pypi.org/project/pyTMD/>`_ (on releases)
