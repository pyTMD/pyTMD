Astronomy
#########

Arguments
---------

The tide potential is a function of the position of the sun and moon with respect to the Earth.
The complete movements of the three bodies in three dimensions are very complicated, and typically require the use of numerical :term:`Ephemerides` :cite:p:`Pugh:2014di`.
:cite:t:`Doodson:1921kt` described the approximate positions in terms of fundamental astronomical arguments.
Each of these arguments can be accurately calculated using polynomial expansions of time :cite:p:`Meeus:1991vh` :cite:p:`Simon:1994vo`.
The rates of change of these arguments are the fundamental frequencies of the astronomical motions :cite:p:`Pugh:2014di` :cite:p:`Kantha:2000vo`.

.. list-table:: Astronomical Arguments
    :header-rows: 1

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
      - lunar perigree
      - 8.847 years
    * - :math:`N`
      - ascending lunar node
      - 18.61 years
    * - :math:`Ps`
      - solar perigree
      - 21,000 years

The lunar hour angle (:math:`\tau`) can be determined from solar time (:math:`t`) using the mean longitudes of the moon (:math:`S`) and sun (:math:`H`):

.. math::
    :label: 4.1
    :name: eq:4.1

    \tau = t - S + H

When calculating :term:`Nutation`, IERS conventions use Delaunay arguments as the fundamental variables :cite:p:`Woolard:1953wp` :cite:p:`Capitaine:2003fx` :cite:p:`Petit:2010tp` .

.. list-table:: Delaunay Arguments
    :header-rows: 1

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
      
These arguments can be calculated from Doodson arguments using the following relationships:

.. math::
    :label: 4.2
    :name: eq:4.2

    \gamma &= \tau + S \\
    l &= S - P \\
    l' &= h - Ps \\
    F &= S - N \\
    D &= S - H \\
    \Omega &= N \\

Nutation
--------

:term:`Nutation` is the periodic oscillation of the Earth's rotation axis around its mean position.
Nutation is often split into two components, the nutation in longitude and the nutation in obliquity.
The angle between the equator and the orbital plane of Earth around the Sun (the :term:`Ecliptic`) defines the inclination of the Earth's rotation axis (obliquity of the ecliptic).
