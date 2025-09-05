Tides
#####

Ocean and Load Tides
--------------------

The rise and fall of the oceanic tides are a major source of the vertical variability of the ocean surface.
Ocean tides are driven by gravitational undulations due to the relative positions of the Earth, moon and sun, and the centripetal acceleration due to the Earth's rotation :cite:p:`Doodson:1921kt,Meeus:1991vh`.
A secondary tidal effect, known as load tides, is due to the elastic response of the Earth's crust to ocean tidal loading, which produces deformation of both the sea floor and adjacent land areas.
Ocean tides can be observed using float gauges, GPS stations, gravimeters, tiltmeters, pressure recorders, and satellite altimeters.

.. note::
    Different measurement techniques can have different `vertical datums <https://www.esr.org/data-products/antarctic_tg_database/ocean-tide-and-ocean-tide-loading/>`_!
    Tide gauges measure the height of the ocean surface relative to the land upon which they are situated (*ocean tides only*).
    Satellite altimeters measure the height of the ocean surface relative to the center of mass of the Earth system (*combination of ocean and earth tides*).

Tidal oscillations for both ocean and load tides can be decomposed into a series of tidal constituents (or partial tides) of particular frequencies that are associated with the relative positions of the sun, moon and Earth.
These tidal constituents are typically classified into different "species" based on their approximate period: short-period, semi-diurnal, diurnal, and long-period [see :ref:`tab-constituents`].

.. plot:: ./background/spectra.py
    :show-source-link: False
    :caption: Tidal spectra from :cite:t:`Cartwright:1973em`
    :align: center

The amplitude and phase of major constituents are provided by ocean tide models, which can be used for tidal predictions.
Ocean tide models are typically one of following categories:
1) empirically adjusted models,
2) barotropic hydrodynamic models constrained by data assimilation, and
3) unconstrained hydrodynamic models :cite:p:`Stammer:2014ci`.

.. note::

    ``pyTMD`` is not an ocean or load tide model, but rather a tool for using constituents from tide models to calculate the height deflections or currents at particular locations and times :cite:p:`Egbert:2002ge`.

Under the equilibrium theory of tides, the Earth is a spherical body with a uniform distribution of water over its surface :cite:p:`Doodson:1921kt`.
In this model, the oceanic surface instantaneously responds to the tide-producing forces of the moon and sun, and is not influenced by inertia, currents or the irregular distribution of land :cite:p:`Schureman:1958ty`.
However in reality, every constituent lags behind its corresponding equilibrium wave, and their amplitudes differ in magnitude :cite:p:`Dronkers:1975hm`.
While the equilibrium condition is rarely satisfied for shorter period tides, some of the longest period ocean tides are often assumed to be well approximated as equilibrium responses to the tidal force :cite:p:`Proudman:1960jj,Ray:2014fu`. 

Using the relative amplitudes from equilibrium theory are also useful for *inferring* unmodeled constituents :cite:p:`Cartwright:1971iz,Cartwright:1973em`.
Tidal inference refers to the estimation of smaller (minor) constituents from estimates of the more major constituents :cite:p:`Ray:2017jx`.
Inference is a useful tool for estimating more of the tidal spectrum when only a limited set of constituents are provided by a tide model :cite:p:`Parker:2007wq`.
For tides in the diurnal band, a resonance from the Earth's free core notation (FCN) can complicate inferring some constituents :cite:p:`Wahr:1981if,Ray:2017jx,Agnew:2018ih`.
This resonance affects the instantaneous elastic response of the solid Earth to tidal loading :cite:p:`Wahr:1979vx`.

``pyTMD.io`` contains routines for reading major constituent values from commonly available tide models, and interpolating those values to spatial locations.
``pyTMD`` uses the astronomical argument formalism outlined in :cite:t:`Doodson:1921kt` for the prediction of ocean and load tides. 
For any given time, ``pyTMD.astro`` calculates the longitudes of the moon (:math:`S`), sun (:math:`H`), lunar perigree (:math:`P`), ascending lunar node (:math:`N`) and solar perigree (:math:`Ps`), which are used in combination with the lunar hour angle (:math:`\tau`) and the extended Doodson number (:math:`k`) in a seven-dimensional Fourier series :cite:p:`Doodson:1921kt,Dietrich:1980ua,Pugh:2014di`.
Each constituent has a particular "Doodson number" describing the polynomial coefficients of each of these astronomical terms in the Fourier series :cite:p:`Doodson:1921kt`. 

.. math::
    :label: 1.1
    :name: eq:1.1

    \sigma(t) = d_1\tau + d_2 S + d_3 H + d_4 P + d_5 N + d_6 Ps + d_7 k

.. tip::

    ``pyTMD`` stores these coefficients in a `JSON database <https://github.com/pyTMD/pyTMD/blob/main/pyTMD/data/doodson.json>`_ supplied with the program.

Together the Doodson coefficients and additional nodal corrections (:math:`f` and :math:`u`) are used by ``pyTMD`` to calculate the frequencies and 18.6-year modulations of the tidal constituents, and enable the accurate determination of tidal values :cite:p:`Schureman:1958ty,Dietrich:1980ua`.
After the determination of the major constituents, ``pyTMD`` can estimate the amplitudes of minor constituents using inference methods :cite:p:`Schureman:1958ty,Ray:2017jx`.


Solid Earth Tides
-----------------

Similar to ocean tides, solid Earth tides (or body tides) are tidal deformations due to gravitational undulations based on the relative positions of the Earth, moon and sun :cite:p:`Agnew:2015kw,Doodson:1921kt,Meeus:1991vh,Montenbruck:1989uk`.
However, while ocean tides are apparent to observers on the coast, solid Earth tides are typically more difficult to observe due to the reference frame of the observer moving.
The tidal deformation of the Earth is to a very high degree instantaneous, with the Earth's response to the gravitational potential of the moon and sun being nearly immediate.
The total gravitational potential at a position on the Earth's surface due to a celestial object is directly related to the distance between the Earth and the object, and the mass of that object :cite:p:`Agnew:2015kw,Wahr:1981ea`.

Within ``pyTMD``, the tidal deformation of the Earth can be modeled using two methods:
1) using ephemerides and the formalism described in the `IERS Conventions <https://iers-conventions.obspm.fr/>`_, which are based on :cite:t:`Wahr:1981ea,Mathews:1997js`, or
2) using tide potential catalogs and the spherical harmonic formalism described in :cite:t:`Cartwright:1971iz`.
For the ephemerides method, analytical approximate positions for the sun and moon can be calculated, or high-resolution numerical ephemerides for the sun and moon can be downloaded from the `Jet Propulsion Laboratory <https://ssd.jpl.nasa.gov/planets/orbits.html>`_.
These astronomical positions are used to estimate the instantaneous tide potential impacting the solid Earth.

For both methods, the elastic response of the Earth to the tidal potential is calculated using :term:`Love and Shida Numbers`.
Love and Shida numbers describe the elastic response of the Earth in terms of vertical displacement (:math:`h`), gravitational potential (:math:`k`) and horizontal displacement (:math:`l`) :cite:p:`Munk:1960uk`.
For a spherical, non-rotating Earth, the Love and Shida numbers are largely independent of tidal frequency :cite:p:`Wahr:1979vx,Wahr:1981ea`.
However, for a rotating, ellipsoidal Earth, the Love and Shida numbers are dependent on tidal frequency, with resonances particularly in the diurnal band :cite:p:`Wahr:1979vx,Wahr:1981ea,Ray:2017jx`.
``pyTMD`` computes these frequency-dependent corrections along with the dissipative mantle anelasticity corrections following :cite:t:`Mathews:1997js`.

.. plot:: ./background/love-numbers.py
    :show-source-link: False
    :caption: Diurnal frequency dependence of :term:`Love and Shida Numbers` from :cite:t:`Wahr:1979vx`
    :align: center

In addition to the ups and downs of tides, there is a considerable portion of tidal potential and displacement that does not vary in time, a *permanent tide* that is due to the Earth being in the presence of the Sun and Moon (and other planetary bodies).
The `Earth is lower in polar areas and higher in equatorial areas <https://www.ngs.noaa.gov/PUBS_LIB/EGM96_GEOID_PAPER/egm96_geoid_paper.html>`_ than it would without those gravitational effects.
The `IERS formalism <https://iers-conventions.obspm.fr/>`_ for determining station locations is to remove all cyclical and permanent components of the tides, which is known as a "tide-free" system.
This is the default "tide-system" within ``pyTMD``.
Alternatively, the permanent tide components can be added back in order to calculate the station locations in a "mean-tide" state.
The radial difference in terms of latitude between the mean-tide and tide-free systems is:

.. math::
    :label: 1.2
    :name: eq:1.2

    \delta r(\varphi) = -0.120582 \left(\frac{3}{2} sin^2 \varphi - \frac{1}{2} \right)


Pole Tides
----------

The Earth's rotation axis is inclined at an angle of 23.5 degrees to the celestial pole, and rotates about it once every 26,000 years :cite:p:`Kantha:2000vo`.
Superimposed on this long-term :term:`Precession`, the rotation axis of the Earth shifts with respect to its mean pole location due to :term:`Nutation`, :term:`Chandler Wobble`, annual variations, and other processes :cite:p:`Wahr:1985gr,Desai:2002ev,Agnew:2015kw`.
Load and ocean pole tides are driven by these variations, the corresponding elastic response, and for the case of ocean pole tides the centripetal effects of :term:`Polar Motion` on the ocean :cite:p:`Desai:2002ev,Desai:2015jr`.
These variations are centimeter scale in both the vertical and horizontal, and should be taken into account when comparing observations over periods longer than two months.

The formalism for estimating the pole tides within ``pyTMD`` is also based upon `IERS Conventions <https://iers-conventions.obspm.fr/>`_.
For ocean pole tides, ``pyTMD`` uses the equilibrium response model from :cite:t:`Desai:2002ev` as recommended by IERS Conventions :cite:p:`Petit:2010tp`.
``pyTMD`` uses the ``timescale`` library for reading the Earth Orientation Parameters (EOPs) necessary for computing load pole and ocean pole tide variations.
The currently accepted formalism for estimating the reference position of the Earth's figure axis at a given date is the `IERS 2018 secular pole model <https://iers-conventions.obspm.fr/chapter7.php>`_:

.. math::
    :label: 1.3
    :name: eq:1.3

    \bar{x}_s(t) &= 0.055 + 0.001677(t - 2000.0)\\
    \bar{y}_s(t) &= 0.3205 + 0.00346(t - 2000.0)


The time-dependent offsets from the reference rotation pole position, also known as wobble parameters (:math:`m_1` and :math:`m_2`), are then calculated using instantaneous values of the Earth Orientation Parameters :cite:p:`Petit:2010tp,Urban:2013vl`.


.. math::
    :label: 1.4
    :name: eq:1.4

    m_1(t) &= x_p(t) - \bar{x}_s(t)\\
    m_2(t) &= -(y_p(t) - \bar{y}_s(t))

.. plot:: ./background/polar-motion.py
    :show-source-link: False
    :caption: Polar motion estimates from the IERS
    :align: center
