:orphan:

.. _jpl-ephemerides:

===============
JPL Ephemerides
===============

``pyTMD`` can read planetary ephemeris data to compute high-resolution positions of the Moon and Sun.
The data come from `Spacecraft and Planet Kernel (SPK) <https://naif.jpl.nasa.gov/pub/naif/toolkit_docs/C/req/spk.html>`_ files storing the positions (and possibly velocities) of solar-system bodies :cite:p:`Park:2021fa`.
The SPK and other binary files are distributed by the NASA Jet Propulsion Laboratory `Navigation and Ancillary Information Facility (NAIF) <https://naif.jpl.nasa.gov/naif/index.html>`_ :cite:p:`Acton:1996jf`.

.. note::

   ``pyTMD`` reads SPK files through the `jplephem <https://pypi.org/project/jplephem/>`_ library.

Development Ephemerides
=======================

Development Ephemeris (DE) kernels are general-purpose SPKs :cite:p:`Folkner:2009wm,Folkner:2014un` useful for high-precision applications.
Each release of a DE kernel incorporates the most recent data from spacecraft tracking, lunar laser ranging, planetary doppler radar, and other observations :cite:p:`Park:2021fa`.

DE "short" kernels are relatively small files designed for high-precision applications over a limited time span.
DE "medium" kernels are larger, but have a similar precision as the short kernels while covering an extended time span (centuries).
DE "long" kernels are much larger files and designed for lower-precision applications over very long time spans (centuries to millennia).

.. list-table:: `DE-series SPK files <https://ssd.jpl.nasa.gov/planets/eph_export.html>`_
   :header-rows: 1
   :align: center
   :widths: 12 28 28 28

   * - Issued
     - Short
     - Medium
     - Long
   * - 1997
     - 
     - ``de405.bsp``
     - ``de406.bsp``
   * - 2008
     - ``de421.bsp``
     - 
     - ``de422.bsp``
   * - 2013
     - ``de430_1850-2150.bsp``
     - ``de430t.bsp``
     - ``de431t.bsp``
   * - 2020
     - ``de440s.bsp``
     - ``de440.bsp``
     - ``de441.bsp``


``DE440s.bsp`` is the default kernel used by ``pyTMD`` as it is a small file covering the modern era.
If locally unavailable, the kernel file is automatically downloaded from the JPL server when calling :py:func:`pyTMD.astro.solar_ephemerides` or :py:func:`pyTMD.astro.lunar_ephemerides`.

.. tip::

    The choice of ephemeris kernel would not likely affect the results of most tidal applications.

Positions in some older SPK files were referenced to Barycentric Dynamical Time (TDB), which slightly differs from terrestrial time (TT) due to relativistic effects [see :ref:`dynamical-time`].
These differences are typically less than 2 milliseconds, and negligible for most tidal applications.

NAIF Codes
==========

SPK files contain *segments* that describe the positions of a *target* body relative to a *center* body over a finite time interval.
The segment data can be queried using specific integer codes called NAIF IDs.
The following NAIF IDs can be used to compute the geocentric positions of the Sun and Moon:

.. list-table::
   :header-rows: 1
   :align: center

   * - Body/System
     - NAIF ID
     - Notes
   * - Solar System Barycenter (SSB)
     - 0
     - Center of mass of the solar system
   * - Earth-Moon Barycenter (EMB)
     - 3
     - Center of mass of the Earth-Moon system
   * - Sun
     - 10
     - Relative to SSB
   * - Moon
     - 301
     - Relative to EMB
   * - Earth
     - 399
     - Relative to EMB

