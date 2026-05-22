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

The angle between the equator and the orbital plane of Earth around the Sun (the :term:`ecliptic <Ecliptic>`) defines the inclination of the Earth's rotation axis (:term:`obliquity <Obliquity>` of the ecliptic).
:term:`Nutations <Nutation>` are the periodic oscillations of the Earth's rotation axis around its mean position, which arise from the time-varying torques exerted on Earth's equatorial bulge :cite:p:`Dehant:2015vb,Woolard:1953wp`.
These largely short-period wobbles have durations ranging from subdaily to multi-annual, which are superimposed on the much slower :term:`precession <Precession>` of the rotation axis :cite:p:`Dehant:2015vb`.
The Earth's nutation is conventionally resolved into two components measured with respect to the :term:`ecliptic <Ecliptic>` :cite:p:`Dehant:2015vb,Meeus:1991vh`:

- **Nutation in longitude** (:math:`\Delta\psi`)\ **:** shift in the position of the true :term:`vernal equinox <Vernal Equinox>` along the ecliptic
- **Nutation in obliquity** (:math:`\Delta\varepsilon`)\ **:** variation in the tilt of Earth's equator with respect to the ecliptic

These two quantities help define the orientation of the true equator and equinox of date relative to the mean equator and :term:`equinox <Equinox>`.
For instance, the instantaneous (true) obliquity of the ecliptic is:

.. math::
    :label: 6.4
    :name: eq:6.4

    \varepsilon = \bar\varepsilon + \Delta\varepsilon

where :math:`\bar\varepsilon` is the mean obliquity :cite:p:`Capitaine:2003fx,Capitaine:2003fw`.
These nutation angles (:math:`\Delta\psi` and :math:`\Delta\varepsilon`) and the mean obliquity (:math:`\bar\varepsilon`) are combined when forming the nutation rotation matrix (:math:`\mathbf{N}`) used in the transformation from celestial to terrestrial reference frames (see :ref:`celestial-reference` and :ref:`Equation 4.5 <eq:4.5>`) :cite:p:`Kaplan:1989cf,Petit:2010tp`.

Equation of the Equinoxes
-------------------------

The difference between Greenwich Apparent Sidereal Time (GAST) and Greenwich Mean Sidereal Time (GMST) defines the "equation of the equinoxes" :math:`\alpha_e`, which is calculated using the :term:`nutation <Nutation>` terms along with small higher-order complementary terms (:math:`e_{comp}`) :cite:p:`Capitaine:2003fx,Capitaine:2003fw,Petit:2010tp`:

.. math::
   :label: eq-eqeq

   \alpha_e = \Delta\psi \cos\varepsilon + e_{comp}

:func:`pyTMD.astro.itrs` calculates GAST when forming the ITRS rotation matrix for converting a celestial reference frame to an Earth-centered Earth-fixed (ECEF) reference frame (see :ref:`celestial-reference` and :ref:`ephemerides`) :cite:p:`Petit:2010tp`.

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
    :label: 6.5
    :name: eq:6.5

    \cos\psi = \sin\varphi\sin\delta + \cos\varphi\cos\delta\cos h

where :math:`\psi` is the zenith angle, :math:`\varphi` is the latitude on the Earth's surface, :math:`\delta` is the declination of the celestial body, and :math:`h` is the local hour angle of the celestial body.
This is equivalent to the dot product between the unit vectors of the celestial body and the position on the Earth's surface.
``pyTMD`` takes advantage of this relationship in order to calculate the zenith angles of the sun and moon using their positions in Cartesian coordinates (see :ref:`ephemerides`).
