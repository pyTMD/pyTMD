Reference Systems
#################

Cartesian coordinate systems are represented by three perpendicular axes that are defined with respect to an origin, and with directions defined for each axis :cite:p:`Urban:2013vl`.
Locations of planetary bodies and satellites can be determined in an Earth-centered Earth-Fixed coordinate system (ECEF) :cite:p:`Meeus:1991vh,Montenbruck:1989uk`.
ECEF is a Cartesian coordinate system with :math:`x`, :math:`y`, and :math:`z` defined with respect to the Earth's center of mass.
The :math:`z` axis is aligned with the Earth's rotation axis, the :math:`x` axis is aligned with the intersection of the prime meridian and the equator, and the :math:`y` axis is aligned with 90\ |degree| east longitude and the equator.
The :math:`xy` plane is also called the equatorial plane.

As opposed to simple vertical offsets, changing the terrestial reference system can involve both `translation and rotation of the reference system <https://itrf.ign.fr/doc_ITRF/Transfo-ITRF2014_ITRFs.txt>`_.
This involves converting from a geographic coordinate system into a Cartesian coordinate system, and then performing matrix transformations.
In general, objects move with respect to the coordinate system, and the coordinate system itself moves and rotates in space :cite:p:`Urban:2013vl`.
Coordinate system definitions, such as the International Terrestrial Reference Frame (ITRF), will often include a time component to account for these changes.

Geodetic coordinates (longitude :math:`\lambda`, latitude :math:`\varphi`, and height :math:`h`) are used to describe the position of a point on the Earth with respect to a defined ellipsoid.
The Cartesian coordinates of a point in space can be calculated from these ellipsoidal coordinates:

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

where :math:`N` is the radius of curvature in the prime vertical [:ref:`Equation 2.2 <eq:2.2>`], :math:`e` is the ellipsoidal eccentricity [:ref:`Equation 2.3 <eq:2.3>`], :math:`a` is the semi-major axis of the ellipsoid, and :math:`f` is the ellipsoidal flattening :cite:p:`HofmannWellenhof:2006hy,Urban:2013vl`.

Geoid Height
------------

The height above mean sea level of a point on the Earth is defined with respect to an irregular surface known as the :term:`Geoid`.
The :term:`Geoid` is the instantaneous shape of the Earth's gravitational field, which would coincide with global mean sea level if the oceans were at rest.
It is an equipotential surface, or a surface of constant potential energy :cite:p:`HofmannWellenhof:2006hy`.
The distance between the geoid and the reference ellipsoid is called the geoid height (:math:`N`) :cite:p:`HofmannWellenhof:2006hy`.

.. figure:: ../_assets/geoid_height.svg
    :width: 400
    :align: center

    Relationship between ellipsoid height, geoid height, and topographic height :cite:p:`NRC:1997ea`

.. |degree|    unicode:: U+00B0 .. DEGREE SIGN
