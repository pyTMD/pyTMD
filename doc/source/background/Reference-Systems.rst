Reference Systems
#################

Cartesian coordinate systems are represented by three perpendicular axes that are defined with respect to an origin, and with directions defined for each axis :cite:p:`Urban:2013vl`.
Locations of planetary bodies and satellites can be determined in an Earth-centered Earth-Fixed coordinate system (ECEF) :cite:p:`Meeus:1991vh,Montenbruck:1989uk`.
ECEF is a Cartesian coordinate system with :math:`x`, :math:`y`, and :math:`z` defined with respect to the Earth's center of mass.
The :math:`z` axis is aligned with the Earth's rotation axis, the :math:`x` axis is aligned with the intersection of the prime meridian and the equator, and the :math:`y` axis is aligned with 90\ |degree| east longitude and the equator.
The :math:`xy` plane is also called the equatorial plane.

Geodetic coordinates (longitude :math:`\lambda`, latitude :math:`\varphi`, and height :math:`h`) are used to describe the position of a point on the Earth with respect to a defined ellipsoid.
Changing the terrestial reference system can involve both translations and rotations of the reference system :cite:p:`Urban:2013vl`.
This involves converting from a geographic coordinate system into a Cartesian coordinate system, and then performing matrix transformations [:ref:`Equation 2.1 <eq:2.1>`].

.. math::
    :label: 2.1
    :name: eq:2.1

    (\lambda, \varphi, h) &\rightarrow (x, y, z) \\
    (x, y, z) &\rightarrow (x', y', z') \\
    (x', y', z') &\rightarrow (\lambda', \varphi', h')

The transformation from ellipsoidal coordinates of a point in space to Cartesian coordinates is calculated by:

.. math::
    :label: 2.2
    :name: eq:2.2

    x &= (N + h)\cos{\varphi}\cos{\lambda}\\
    y &= (N + h)\cos{\varphi}\sin{\lambda}\\
    z &= \left((1 - e^2) N + h \right) \sin{\varphi}

.. math::
    :label: 2.3
    :name: eq:2.3

    N = \frac{a}{\sqrt{1 - e^2 \sin^2{\varphi}}}

.. math::
    :label: 2.4
    :name: eq:2.4

    e = 2f - f^2

where :math:`N` is the radius of curvature in the prime vertical [:ref:`Equation 2.3 <eq:2.3>`], :math:`e` is the ellipsoidal eccentricity [:ref:`Equation 2.4 <eq:2.4>`], :math:`a` is the semi-major axis of the ellipsoid, and :math:`f` is the ellipsoidal flattening :cite:p:`HofmannWellenhof:2006hy,Urban:2013vl`.
Ellipsoid definitions typically specify the semi-major axis (:math:`a`) and flattening (:math:`f`), and datum definitions additionally include the coordinate system origin :cite:p:`HofmannWellenhof:2006hy,Urban:2013vl`.
In general, objects move with respect to the coordinate system, and the coordinate system itself moves and rotates in space :cite:p:`Urban:2013vl`.
Coordinate system definitions, such as the International Terrestrial Reference Frame (ITRF), will often include a time component to account for these changes.

Celestial Reference Systems
---------------------------

Celestial reference systems are used to describe the positions of celestial bodies in the sky.
Transforming between celestial (:math:`\mathbf{x}_{CRS}`) and terrestrial (:math:`\mathbf{x}_{TRS}`) reference systems involves a set of transformation matrices for frame bias (:math:`\mathbf{B}`), precession (:math:`\mathbf{P}`), nutation (:math:`\mathbf{N}`), Earth's rotation (:math:`\mathbf{T}`), and polar motion (:math:`\mathbf{W}`) :cite:p:`Capitaine:2003fx,Capitaine:2003fw,Urban:2013vl`.

.. math::
    :label: 2.5
    :name: eq:2.5

    \mathbf{x}_{CRS} = \mathbf{B}\ \mathbf{P}\ \mathbf{N}\ \mathbf{T}\ \mathbf{W}\ \mathbf{x}_{TRS}

In ``pyTMD``, these transformations are used to convert planetary :term:`Ephemerides` from a celestial reference frame to a terrestrial reference frame.

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
