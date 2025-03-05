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
    * - :math:`\tau`
      - lunar hour angle
    * - :math:`S`
      - mean longitude of the moon
    * - :math:`H`
      - mean longitude of the sun
    * - :math:`P`
      - lunar perigree
    * - :math:`N`
      - ascending lunar node
    * - :math:`Ps`
      - solar perigree

The lunar hour angle (:math:`\tau`) can be determined from solar time (:math:`t`) using the mean longitudes of the moon (:math:`S`) and sun (:math:`H`) :cite:p:`Kantha:2000vo`.

.. math::
    :label: 4.1
    :name: eq:4.1

    \tau = t - S + H

Nutation
--------

:term:`Nutation` is the periodic oscillation of the Earth's rotation axis around its mean position.
Nutation is often split into two components, the nutation in longitude and the nutation in obliquity.
The angle between the equator and the orbital plane of Earth around the Sun (the :term:`Ecliptic`) defines the inclination of the Earth's rotation axis (obliquity of the ecliptic).
