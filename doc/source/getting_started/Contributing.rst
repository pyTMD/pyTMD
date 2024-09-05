=======================
Contribution Guidelines
=======================

We welcome and invite contributions from anyone at any career stage and with any amount of coding experience towards the development of ``pyTMD``.
We appreciate any and all contributions made to the project.
Please read our `code of conduct <./Code-of-Conduct.html>`_ before contributing to ``pyTMD`` development.
You will be recognized for your work by being listed as one of the `project contributors <./Citations.html#contributors>`_.

Ways to Contribute
------------------

1) Fixing typographical or coding errors
2) Submitting bug reports or feature requests through the use of `GitHub issues <https://github.com/tsutterley/pyTMD/issues>`_
3) Improving documentation and testing
4) Sharing use cases and examples (such as `Jupyter Notebooks <../user_guide/Examples.html>`_)
5) Providing code for everyone to use
6) Adding model providers to the database

Requesting a Feature
--------------------
Check the `project issues tab <https://github.com/tsutterley/pyTMD/issues>`_ to see if the feature has already been suggested.
If not, please submit a new issue describing your requested feature or enhancement .
Please give your feature request both a clear title and description.
Let us know if this is something you would like to contribute to ``pyTMD`` in your description as well.

Reporting a Bug
---------------
Check the `project issues tab <https://github.com/tsutterley/pyTMD/issues>`_ to see if the problem has already been reported.
If not, *please* submit a new issue so that we are made aware of the problem.
Please provide as much detail as possible when writing the description of your bug report.
Providing information and examples will help us resolve issues faster.

Contributing Code or Examples
-----------------------------
We follow a standard Forking Workflow for code changes and additions.
Submitted code goes through the pull request process for `continuous integration (CI) testing <./Contributing.html#continuous-integration>`_ and comments.

General Guidelines
^^^^^^^^^^^^^^^^^^

- Make each pull request as small and simple as possible
- `Commit messages should be clear and describe the changes <./Contributing.html#semantic-commit-messages>`_
- Larger changes should be broken down into their basic components and integrated separately
- Bug fixes should be their own pull requests with an associated `GitHub issue <https://github.com/tsutterley/pyTMD/issues>`_
- Write a descriptive pull request message with a clear title
- Be patient as reviews of pull requests take time

Steps to Contribute
^^^^^^^^^^^^^^^^^^^

1) Fork the repository to your personal GitHub account by clicking the "Fork" button on the project `main page <https://github.com/tsutterley/pyTMD>`_.  This creates your own server-side copy of the repository.
2) Either by cloning to your local system or working in `GitHub Codespaces <https://github.com/features/codespaces>`_, create a work environment to make your changes.
3) Add your fork as the ``origin`` remote and the original project repository as the ``upstream`` remote.  While this step isn't a necessary, it allows you to keep your fork up to date in the future.
4) Create a new branch to do your work.
5) Make your changes on the new branch and add yourself to the list of project `contributors <https://github.com/tsutterley/pyTMD/blob/main/CONTRIBUTORS.rst>`_.
6) Push your work to GitHub under your fork of the project.
7) Submit a `Pull Request <https://github.com/tsutterley/pyTMD/pulls>`_ from your forked branch to the project repository.

Adding Examples
^^^^^^^^^^^^^^^
Examples may be in the form of executable scripts or interactive `Jupyter Notebooks <../user_guide/Examples.html>`_.
Fully working (but unrendered) examples should be submitted with the same steps as above.

Adding Model Providers
^^^^^^^^^^^^^^^^^^^^^^
Model providers can be added to their respective `JSON files <https://github.com/tsutterley/pyTMD/tree/main/providers>`_.
The main database can then be updated by running the `merge providers <https://github.com/tsutterley/pyTMD/blob/main/providers/_providers_to_database.py>`_ script.

Continuous Integration
^^^^^^^^^^^^^^^^^^^^^^
We use `GitHub Actions <https://github.com/tsutterley/pyTMD/actions>`_ continuous integration (CI) services to build and test the project on Linux (Ubuntu) and Mac Operating Systems.
The configuration files for this service are in the `GitHub workflows <https://github.com/tsutterley/pyTMD/tree/main/.github/workflows>`_ directory.
The workflows rely on the `environment.yml <https://github.com/tsutterley/pyTMD/blob/main/environment.yml>`_ and `requirements-dev.txt <https://github.com/tsutterley/pyTMD/blob/main/requirements-dev.txt>`_ files to install the required dependencies.

The GitHub Actions jobs include:

* Running `flake8 <https://flake8.pycqa.org/en/latest/>`_ to check the code for style and compilation errors
* Running the test suite on multiple combinations of OS and Python version
* Uploading test coverage statistics to `Codecov <https://app.codecov.io/gh/tsutterley/pyTMD>`_
* Uploading source and wheel distributions to `PyPI <https://pypi.org/project/pyTMD/>`_ (on releases)

Semantic Commit Messages
^^^^^^^^^^^^^^^^^^^^^^^^

Please follow the `Conventional Commits <https://www.conventionalcommits.org/>`_ specification for your commit messages to help organize the pull requests:

.. code-block:: bash

    <type>: <subject>

    [optional message body]

where ``<type>`` is one of the following:

- ``feat``: adding new features or programs
- ``fix``: fixing bugs or problems
- ``docs``: changing the documentation
- ``style``: changing the line order or adding comments
- ``refactor``: changing the names of variables or programs
- ``ci``: changing the `continuous integration <./Contributing.html#continuous-integration>`_ configuration files or scripts
- ``test``: adding or updating `continuous integration tests <./Contributing.html#continuous-integration>`_
