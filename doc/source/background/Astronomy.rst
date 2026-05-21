Astronomy
#########

Arguments
---------

The tide potential is a function of the position of the sun and moon with respect to the Earth.
The complete movements of the three bodies in three dimensions are very complicated, and typically require the use of numerical :term:`ephemerides <Ephemerides>` :cite:p:`Pugh:2014di`.
:cite:t:`Doodson:1921kt` described the approximate positions in terms of fundamental astronomical arguments.
Each of these arguments can be accurately calculated using polynomial expansions of time :cite:p:`Meeus:1991vh,Simon:1994vo`.
The rates of change of these arguments are the fundamental frequencies of the astronomical motions :cite:p:`Pugh:2014di,Kantha:2000vo`.

.. list-table:: Astronomical Arguments
    :header-rows: 1
    :align: center

    * - Argument
      - Description
      - Period
    * - :math:`\tau`
      - lunar hour angle
      - 1.03505 days
    * - :math:`S`
      - mean longitude of the moon
      - 27.32158 days
    * - :math:`H`
      - mean longitude of the sun
      - 365.2549 days
    * - :math:`P`
      - lunar perigee
      - 8.847 years
    * - :math:`N`
      - ascending lunar node
      - 18.61 years
    * - :math:`Ps`
      - solar perigee
      - 21,000 years

The lunar hour angle (:math:`\tau`) can be determined from solar time (:math:`t`) using the mean longitudes of the moon (:math:`S`) and sun (:math:`H`):

.. math::
    :label: 6.1
    :name: eq:6.1

    \tau = t - S + H

When calculating :term:`nutation <Nutation>`, IERS conventions use Delaunay arguments as the fundamental orbital elements :cite:p:`Woolard:1953wp,Capitaine:2003fx,Petit:2010tp` .

.. list-table:: Delaunay Arguments
    :header-rows: 1
    :align: center

    * - Argument
      - Description
      - Period
    * - :math:`\gamma`
      - mean sidereal time
      - 0.99727 days
    * - :math:`l`
      - mean anomaly of the moon
      - 27.5545 days
    * - :math:`l'`
      - mean anomaly of the sun
      - 365.2596 days
    * - :math:`F`
      - mean argument of latitude of the moon
      - 27.2122 days
    * - :math:`D`
      - mean elongation of the moon from the sun
      - 29.5306 days
    * - :math:`\Omega`
      - ascending lunar node
      - 18.61 years
      
From :cite:t:`Dehant:2015vb`, these arguments can be calculated from Doodson arguments using the following relationships:

.. math::
    :label: 6.2
    :name: eq:6.2

    \gamma &= \tau + S \\
    l &= S - P \\
    l' &= h - Ps \\
    F &= S - N \\
    D &= S - H \\
    \Omega &= N = -N' \\

And conversely:

.. math::
    :label: 6.3
    :name: eq:6.3

    S &= F + \Omega \\
    H &= F + \Omega - D \\
    P &= F + \Omega - l\\
    N &= \Omega = -N' \\
    Ps &= F + \Omega - l' - D

Nutation
--------

:term:`Nutation` is the periodic oscillation of the Earth's rotation axis around its mean position.
Nutation is often split into two components, the nutation in longitude and the nutation in obliquity.
The angle between the equator and the orbital plane of Earth around the Sun (the :term:`ecliptic <Ecliptic>`) defines the inclination of the Earth's rotation axis (:term:`obliquity <Obliquity>` of the ecliptic).

.. _ephemerides:

Ephemerides
-----------

``pyTMD`` can calculate the positions of the sun and moon relative to the Earth using approximate relations (see :func:`pyTMD.astro.solar_approximate` and :func:`pyTMD.astro.lunar_approximate`), or use the ``jplephem`` package to read `JPL Ephemerides <https://ssd.jpl.nasa.gov/planets/orbits.html>`_ (see :func:`pyTMD.astro.solar_ephemerides` and :func:`pyTMD.astro.lunar_ephemerides`).
Ephemerides are tables of values that give the positions of astronomical objects at a given time.


.. _zenith-angle:

Zenith Angles
-------------

The :term:`zenith angles <Zenith Angle>` of the sun and moon are important for calculating the total tidal potentials, as they determine the position of the celestial body relative to a position on the Earth's surface.

.. math::
    :label: 6.4
    :name: eq:6.4

    \cos\psi = \sin\varphi\sin\delta + \cos\varphi\cos\delta\cos h

where :math:`\psi` is the zenith angle, :math:`\varphi` is the latitude on the Earth's surface, :math:`\delta` is the declination of the celestial body, and :math:`h` is the local hour angle of the celestial body.
This is equivalent to the dot product between the unit vectors of the celestial body and the position on the Earth's surface.
``pyTMD`` takes advantage of this relationship in order to calculate the zenith angles of the sun and moon using their positions in Cartesian coordinates (see :ref:`ephemerides`).
