Ocean and Load Tides
####################

The rise and fall of the oceanic tides are a major source of the vertical variability of the ocean surface.
Ocean tides are driven by gravitational undulations due to the relative positions of the Earth, moon and sun, and the centripetal acceleration due to the Earth's rotation :cite:p:`Doodson:1921kt,Meeus:1991vh`.
A secondary tidal effect, known as load tides, is due to the elastic response of the Earth's crust to ocean tidal loading, which produces deformation of both the sea floor and adjacent land areas.
Ocean tides can be observed using float gauges, GPS stations, gravimeters, tiltmeters, pressure recorders, and satellite altimeters.

.. note::
    Different measurement techniques can have different `vertical datums <https://www.esr.org/data-products/antarctic_tg_database/ocean-tide-and-ocean-tide-loading/>`_!
    Tide gauges measure the height of the ocean surface relative to the land upon which they are situated (*ocean tides only*).
    Satellite altimeters measure the height of the ocean surface relative to the center of mass of the Earth system (*combination of ocean and earth tides*).

Harmonic Method
---------------

``pyTMD`` uses the harmonic method for tide prediction, which is based on harmonic analysis and decomposition.
Here, tidal oscillations for both ocean and load tides are decomposed into a series of tidal constituents (or partial tides) of particular frequencies.
These frequencies are associated with particular astronomical forcings, which are largely based on the relative positions of the sun, moon and Earth.
The tide height (or current) at any time can be estimated through a summation of all available tidal constituents :cite:p:`Doodson:1921kt,Foreman:1989dt`:

.. math::
    :label: 1.1
    :name: eq:1.1

    h(t) = z_0 +\sum_{k=1}^{n} f_k(t) \left[A_k\cos{\left(G_k(t) + u_k(t) + \theta_k\right)} \right]

where :math:`z_0` is the datum offset, :math:`k` is the constituent number, :math:`n` is the total number of considered constituents, :math:`A_k` and :math:`\theta_k` are the constituent amplitude and phase lag *provided by the tide model*, :math:`G_k` is the equilibrium phase [see :ref:`Equation 1.2 <eq:1.2>`], :math:`f_k(t)` and :math:`u_k(t)` are the nodal amplitude and phase modulations [see :term:`Nodal Corrections`].
Tidal constituents are typically classified into different "species" based on their approximate period: short-period, semi-diurnal, diurnal, and long-period [see :ref:`tab-constituents` and :ref:`fig-sphharm`].

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

Nodal Modulations
-----------------

The moon's orbital plane precesses (rotates) in space and completes one revolution approximately every 18.6 years :cite:p:`Dronkers:1975hm`.
This precession causes the maximum declination of the moon to vary between approximately 18 and 28 degrees, with the average equal to the Earth's equatorial inclination (approximately 23 degrees) :cite:p:`Schureman:1958ty`.
When the moon is at its maximum declination, the difference in tide potential causes the (lunar) diurnal tides to be at their largest.
Conversely, when the moon is at its minimum declination, the difference causes the (lunar) semi-diurnal tides to be at their largest.
The moon's perigee also varies with a period of approximately 8.8 years, which causes (smaller) modulations in tidal amplitudes :cite:p:`Dronkers:1975hm`.
Both of these modulations need to be taken into account to properly predict the tidal amplitudes at a given time, or to solve for tidal constituents from long-term observations [see :term:`Nodal Corrections`].

Equilibrium Theory
------------------

Under the equilibrium theory of tides, the Earth is a spherical body with a uniform distribution of water over its surface :cite:p:`Doodson:1921kt`.
In this model, the oceanic surface instantaneously responds to the tide-producing forces of the moon and sun, and is not influenced by inertia, currents or the irregular distribution of land :cite:p:`Schureman:1958ty`.
However in reality due to a combination of hydrodynamic and real world effects, every constituent lags behind its corresponding equilibrium wave, and their amplitudes differ in magnitude :cite:p:`Dronkers:1975hm`.
While the equilibrium condition is rarely satisfied for shorter period tides, some of the longest period ocean tides are often assumed to be well approximated as equilibrium responses to the tidal force :cite:p:`Proudman:1960jj,Ray:2014fu`. 

Using the relative amplitudes from equilibrium theory are also useful for *inferring* unmodeled constituents :cite:p:`Cartwright:1971iz,Cartwright:1973em`.
Tidal inference refers to the estimation of smaller (minor) constituents from estimates of the more major constituents :cite:p:`Ray:2017jx`.
Inference is a useful tool for estimating more of the tidal spectrum when only a limited set of constituents are provided by a tide model :cite:p:`Parker:2007wq`.
For tides in the diurnal band, a resonance from the Earth's free core nutation (FCN) can complicate inferring some constituents :cite:p:`Wahr:1981if,Ray:2017jx,Agnew:2018ih`.
This resonance affects the instantaneous elastic response of the solid Earth to tidal loading :cite:p:`Wahr:1979vx`.

Prediction
----------

``pyTMD.io`` contains routines for reading major constituent values from commonly available tide models, and interpolating those values to spatial locations.
``pyTMD`` uses the astronomical argument formalism outlined in :cite:t:`Doodson:1921kt` for the prediction of ocean and load tides. 
For any given time, :func:`pyTMD.astro.mean_longitudes` calculates the longitudes of the moon (:math:`S`), sun (:math:`H`), lunar perigee (:math:`P`), ascending lunar node (:math:`N`) and solar perigee (:math:`Ps`), which are used in combination with the lunar hour angle (:math:`\tau`) and the extended Doodson number (:math:`k`) in a seven-dimensional Fourier series :cite:p:`Doodson:1921kt,Dietrich:1980ua,Pugh:2014di`.
Each constituent has a particular "Doodson number" describing the polynomial coefficients of each of these astronomical terms in the Fourier series :cite:p:`Doodson:1921kt`. 
These can be summed together to estimate the equilibrium phase (:math:`G`).

.. math::
    :label: 1.2
    :name: eq:1.2

    G(t) = d_1\tau + d_2 S + d_3 H + d_4 P + d_5 N + d_6 Ps + d_7 k

.. tip::

    ``pyTMD`` stores these coefficients in a `JSON database <https://github.com/pyTMD/pyTMD/blob/main/pyTMD/data/doodson.json>`_ supplied with the program.

Together the Doodson coefficients and additional nodal corrections (:math:`f` and :math:`u`) are used by ``pyTMD`` to calculate the frequencies and 18.6-year modulations of the tidal constituents, and enable the accurate determination of tidal values :cite:p:`Schureman:1958ty,Dietrich:1980ua`.
After the determination of the major constituents, :func:`pyTMD.predict.infer_minor` can estimate the amplitudes of minor constituents using inference methods :cite:p:`Schureman:1958ty,Ray:2017jx`.

