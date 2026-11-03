.. _solid-earth-tides:

=================
Solid Earth Tides
=================

Similar to ocean tides, solid Earth tides (or body tides) are deformations of the Earth caused by changes in the gravitational potential of the Moon, Sun and planets :cite:p:`Agnew:2015kw,Doodson:1921kt,Meeus:1991vh,Montenbruck:1989uk`.
While ocean tides are apparent to an observer on the coast, solid Earth tides move the observer along with the ground, and so can be more difficult to observe directly.
Any technique that measures position relative to the center of mass of the Earth (such as GNSS, VLBI, SLR, DORIS and satellite altimetry) will need to account for the tidal deformation of the Earth.
This tidal deformation is considered to be largely elastic and to a very high degree instantaneous.

The total gravitational potential at a position on the Earth's surface due to a celestial object is directly related to the distance between the Earth and the object, and the mass of that object :cite:p:`Agnew:2015kw,Wahr:1981ea`.
As the Earth rotates, points on the surface experience a changing potential, and the gradient of this potential is the tide-generating force.

.. plot:: ./background/tidal-forces.py
    :caption: Tide-generating force and equilibrium deformation
    :align: center

The elastic response of the Earth to the tidal potential is calculated using :ref:`love-and-shida-numbers` [:ref:`Equation 2.1 <eq:2.1>`].
:term:`Love/Shida numbers <Love and Shida Numbers>` describe the elastic response of the Earth in terms of vertical displacement (:math:`h`), gravitational potential (:math:`k`) and horizontal displacement (:math:`l`) :cite:p:`Lambeck:1980ic,Munk:1960uk`.

.. math::
    :label: 2.1
    :name: eq:2.1

    S_r &= \frac{1}{g}\sum_{l=2}^{\infty} h_l V_l \\
    S_\varphi &= \frac{1}{g}\sum_{l=2}^{\infty} l_l \frac{\partial V_l}{\partial\varphi} \\
    S_\lambda &= \frac{1}{g}\sum_{l=2}^{\infty} l_l \frac{\partial V_l}{\sin\varphi\partial\lambda}

Methods
=======

Within ``pyTMD``, the tidal deformation of the Earth can be modeled using two methods:

1) :py:func:`pyTMD.predict.solid_earth_tide` uses :term:`ephemerides <Ephemerides>` and the formalism described in the `IERS Conventions <https://iers-conventions.obspm.fr/>`_, which are based on :cite:t:`Wahr:1981ea` and :cite:t:`Mathews:1997js`
2) :py:func:`pyTMD.predict.body_tide` uses tide potential catalogs :cite:p:`Wenzel:1997kn` and the spherical harmonic formalism described in :cite:t:`Cartwright:1971iz`.

For the ephemerides method, analytical approximate positions for the Sun and Moon can be calculated, or high-resolution numerical ephemerides for the Sun and Moon can be downloaded from the `Jet Propulsion Laboratory <https://ssd.jpl.nasa.gov/planets/orbits.html>`_.
These astronomical positions are used to estimate the instantaneous tide potential impacting the solid Earth :cite:p:`Merriam:1992kg`.

Most tide potential catalogs have for each tidal constituent, the :ref:`spherical harmonic <spherical-harmonics>` degree and order, the :ref:`Doodson arguments <astronomical-arguments>` and the potential amplitude :cite:p:`Cartwright:1971iz`.
Some catalogs additionally include the potentials induced by the motions of the closest planetary bodies [see :ref:`tab-catalogs`] and higher degree harmonics [see :ref:`spherical-harmonics`].


Ephemerides Method
------------------

From :cite:t:`Mathews:1997js`, the steps for using the ephemerides method to calculate the solid Earth tide displacements at the location and times of interest are:

1) Compute positions of the Sun and Moon in ECEF coordinates (:math:`X`, :math:`Y` and :math:`Z`) [see :ref:`ephemerides`]
2) Calculate frequency-independent deformations using "nominal" :term:`Love/Shida number <Love and Shida Numbers>` values
3) Include contributions from the :term:`Love/Shida number <Love and Shida Numbers>` out-of-phase (imaginary) components
4) Include contributions from the :term:`Love/Shida number <Love and Shida Numbers>` latitudinal dependency for degree 2
5) Include contributions from the :term:`Love/Shida numbers <Love and Shida Numbers>` frequency dependence
6) Convert the :term:`permanent tide system <Permanent Tide>` (if necessary)

Catalog Method
--------------

Similarly, from :cite:t:`Cartwright:1971iz`, the steps for using the catalog method are:

1) Read the tide potential catalog [see :ref:`tab-catalogs`]
2) Calculate the :ref:`astronomical-arguments` and potentially the planetary arguments
3) For each constituent potential in the catalog:

    a) Calculate the angular frequency (:math:`\Omega`) of the constituent
    b) Calculate the frequency and latitude dependent :term:`Love/Shida numbers <Love and Shida Numbers>`
    c) Calculate the :ref:`spherical harmonics <spherical-harmonics>` for the degree and order of the constituent
    d) Calculate the equilibrium phase of the constituent [see :ref:`equilibrium-theory`]
    e) Rotate the spherical harmonics by the equilibrium phase [see :ref:`spherical-harmonics`]
    f) Scale the rotated spherical harmonics by the constituent amplitude and the appropriate :term:`Love/Shida numbers <Love and Shida Numbers>` [see :ref:`Equation 2.1 <eq:2.1>`] to calculate the induced elastic deformation
    g) Add contributions to the total tidal displacement

.. _permanent-tide:

Permanent Tide
==============

In addition to the ups and downs of tides, there is a considerable portion of tidal potential and displacement that does not vary in time.
This ":term:`permanent tide <Permanent Tide>`" exists because the Earth is always in the presence of the Sun and Moon.
The result is that the `Earth is lower in polar areas and higher in equatorial areas <https://www.ngs.noaa.gov/PUBS_LIB/EGM96_GEOID_PAPER/egm96_geoid_paper.html>`_ than it would be without those gravitational effects.
The `IERS formalism <https://iers-conventions.obspm.fr/>`_ for determining station locations is to remove all cyclical and permanent components of the tides, which is known as a ":term:`tide-free <Tide-Free>`" system.
Alternatively, the permanent tide components can be added back in order to calculate the station locations in a ":term:`mean tide <Mean Tide>`" state.
The radial difference in terms of latitude between the ":term:`mean tide <Mean Tide>`" and ":term:`tide-free <Tide-Free>`" systems is:

.. math::
    :label: 2.2
    :name: eq:2.2

    \delta r(\varphi) = -0.120582 \left(\frac{3}{2} \sin^2 \varphi - \frac{1}{2} \right)

.. list-table::
    :header-rows: 1
    :align: center

    * - System
      - Permanent direct potential
      - Permanent deformation
    * - :term:`Tide-free <Tide-Free>`
      - Removed
      - Removed
    * - :term:`Mean tide <Mean Tide>`
      - Kept
      - Kept
    * - :term:`Zero tide <Zero Tide>`
      - Removed
      - Kept


.. warning::

    Mixing tide systems is a common source of error!
    Check which system your reference coordinates or heights use before applying any correction.
