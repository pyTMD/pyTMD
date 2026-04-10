Solid Earth Tides
#################

Similar to ocean tides, solid Earth tides (or body tides) are tidal deformations due to gravitational undulations based on the relative positions of the Earth, moon and sun :cite:p:`Agnew:2015kw,Doodson:1921kt,Meeus:1991vh,Montenbruck:1989uk`.
However, while ocean tides are apparent to observers on the coast, solid Earth tides are typically more difficult to observe due to the reference frame of the observer moving.
The tidal deformation of the Earth is to a very high degree instantaneous, with the Earth's response to the gravitational potential of the moon and sun being nearly immediate.
The total gravitational potential at a position on the Earth's surface due to a celestial object is directly related to the distance between the Earth and the object, and the mass of that object :cite:p:`Agnew:2015kw,Wahr:1981ea`.

Methods
-------

Within ``pyTMD``, the tidal deformation of the Earth can be modeled using two methods:
1) :func:`pyTMD.predict.solid_earth_tide` uses :term:`Ephemerides` and the formalism described in the `IERS Conventions <https://iers-conventions.obspm.fr/>`_, which are based on :cite:t:`Wahr:1981ea` and :cite:t:`Mathews:1997js`, or
2) :func:`pyTMD.predict.body_tide` uses tide potential catalogs :cite:p:`Wenzel:1997kn` and the spherical harmonic formalism described in :cite:t:`Cartwright:1971iz`.
For the ephemerides method, analytical approximate positions for the sun and moon can be calculated, or high-resolution numerical ephemerides for the sun and moon can be downloaded from the `Jet Propulsion Laboratory <https://ssd.jpl.nasa.gov/planets/orbits.html>`_.
These astronomical positions are used to estimate the instantaneous tide potential impacting the solid Earth :cite:p:`Merriam:1992kg`.
For the catalog method, some tide potential catalogs additionally include the potentials induced by the motions of the closest planetary bodies [see :ref:`tab-catalogs`] and higher degree harmonics [see :ref:`fig-sphharm`].

Love and Shida Numbers
----------------------

For both methods, the elastic response of the Earth to the tidal potential is calculated using :term:`Love and Shida Numbers`.
Love and Shida numbers describe the elastic response of the Earth in terms of vertical displacement (:math:`h`), gravitational potential (:math:`k`) and horizontal displacement (:math:`l`) :cite:p:`Munk:1960uk`.
Combinations of these non-dimensional quantities can be used to calculate additional parameters, such as the displacement of the Earth's ocean surface with respect to the Earth's tidally deformed crust :cite:p:`Baker:1984tq,Munk:1960uk`.
For a spherical, non-rotating Earth, the Love and Shida numbers are largely independent of tidal frequency as the tidal periods are longer than the Earth's free oscillation periods :cite:p:`Baker:1984tq,Wahr:1979vx,Wahr:1981ea`.
However, for a rotating, ellipsoidal Earth, the Love and Shida numbers have some dependence on tidal frequency, with resonances particularly in the diurnal band :cite:p:`Wahr:1979vx,Wahr:1981ea,Merriam:1992kg,Ray:2017jx`.
``pyTMD`` computes these frequency-dependent corrections along with the dissipative mantle anelasticity corrections following :cite:t:`Mathews:1997js` and :cite:t:`Wahr:1981ea`.

.. plot:: ./background/love-numbers.py
    :show-source-link: False
    :caption: Diurnal frequency dependence of :term:`Love and Shida Numbers` from :cite:t:`Wahr:1979vx`
    :align: center

Permanent Tide
--------------

In addition to the ups and downs of tides, there is a considerable portion of tidal potential and displacement that does not vary in time, a ":term:`Permanent Tide`" that is due to the Earth being in the presence of the Sun and Moon (and other planetary bodies).
The `Earth is lower in polar areas and higher in equatorial areas <https://www.ngs.noaa.gov/PUBS_LIB/EGM96_GEOID_PAPER/egm96_geoid_paper.html>`_ than it would without those gravitational effects.
The `IERS formalism <https://iers-conventions.obspm.fr/>`_ for determining station locations is to remove all cyclical and permanent components of the tides, which is known as a ":term:`Tide-Free`" system.
This is the default "tide-system" within ``pyTMD``.
Alternatively, the permanent tide components can be added back in order to calculate the station locations in a ":term:`Mean Tide`" state.
The radial difference in terms of latitude between the mean-tide and tide-free systems is:

.. math::
    :label: 2.1
    :name: eq:2.1

    \delta r(\varphi) = -0.120582 \left(\frac{3}{2} \sin^2 \varphi - \frac{1}{2} \right)

