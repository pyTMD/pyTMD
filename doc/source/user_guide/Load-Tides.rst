.. _load-tides:

==========
Load Tides
==========

.. important::
  Running these example recipes requires an :ref:`aws installation <development-install>` to include the optional dependencies.

``pyTMD`` includes two methods for computing load tides from ocean tide constituents.
The spherical harmonic approach is a classic method that works well for global geographic models.
The Green's function approach can be slower, but should work well for regional or projected models.
Both use tables of :ref:`load-love-numbers` to estimate the Earth's elastic response to variations in ocean loading.

Spherical Harmonic Approach
===========================

.. include:: ./harmonic-summation.py
    :literal:

Green's Function Approach
=========================

.. include:: ./greens-functions.py
    :literal:
