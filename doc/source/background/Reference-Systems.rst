Reference Systems
#################

Locations of planetary bodies and satellites can be determined in an Earth-centered Earth-Fixed (ECEF) coordinate system :cite:p:`Montenbruck:1989uk`.
ECEF is a Cartesian coordinate system representing :math:`x`, :math:`y`, and :math:`z` measurements from the Earth's center of mass.
The :math:`z` axis is aligned with the Earth's rotation axis, the :math:`x` axis is aligned with the intersection of the prime meridian and the equator, and the :math:`y` axis is aligned with 90 degrees east longitude and the equator.

The Cartesian coordinates of a point in space can be obtained from its ellipsoidal coordinates:

.. math::
    :label: 2.1
    :name: eq:2.1

    x &= (N + h)\cos{\varphi}\cos{\lambda}\\
    y &= (N + h)\cos{\varphi}\sin{\lambda}\\
    z &= \left((1 - e^2) N + h \right) \sin{\varphi}

.. math::
    :label: 2.2
    :name: eq:2.2

    N = \frac{a}{\sqrt{1 - e^2 \sin^2{\varphi}}}

.. math::
    :label: 2.3
    :name: eq:2.3

    e = 2f - f^2

where :math:`N` is the radius of curvature in the prime vertical [:ref:`Equation 2.2 <eq:2.2>`], :math:`h` is the height above the ellipsoid, :math:`e` is the ellipsoidal eccentricity [:ref:`Equation 2.3 <eq:2.3>`], :math:`a` is the semi-major axis of the ellipsoid, and :math:`f` is the ellipsoidal flattening :cite:p:`HofmannWellenhof:2006hy`.

As opposed to simple vertical offsets, changing the terrestial reference system can involve both `translation and rotation of the reference system <https://itrf.ign.fr/doc_ITRF/Transfo-ITRF2014_ITRFs.txt>`_.
This involves converting from a geographic coordinate system into a Cartesian coordinate system.
Within ``pyTMD``, solid Earth tides are calculated using ECEF coordinates, and pole tides are calculated using geocentric coordinates.

Nutation is the periodic oscillation of the Earth's rotation axis around its mean position.
Nutation is often split into two components, the nutation in longitude and the nutation in obliquity.
The angle between the equator and the orbital plane of Earth around the Sun (the ecliptic) defines the inclination of the Earth's rotation axis (obliquity of the ecliptic).
