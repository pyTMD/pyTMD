:orphan:

.. _fig-sphharm:

Spherical Harmonics
-------------------

The tide potential at colatitude :math:`\theta` and longitude :math:`\phi` is often expressed in terms of spherical harmonic functions :math:`Y_l^m(\theta,\phi)` of degree :math:`l` and order :math:`m` :cite:p:`Doodson:1921kt`.
The degree 2 spherical harmonic terms are the dominant source of tidal excitation, and induce the semi-diurnal (from :math:`Y_2^2`), diurnal (from :math:`Y_2^1`) and long-period (from :math:`Y_2^0`) tides :cite:p:`Ray:2020gn`.

.. plot:: ./background/sphharm.py
    :show-source-link: False
    :caption: Spherical harmonics for degree 2
    :align: center

Global asymmetry in the tide potential can lead to a dependence on higher degree harmonics, most notably the degree 3 and 4 terms.
These terms are small compared to those of degree 2, but have been detected at both local and global scales :cite:p:`Ray:2020gn`.
Both GNSS stations and superconducting gravimeters can be sensitive enough to detect the signals from these higher degree terms :cite:p:`Hartmann:1995jp,Petit:2010tp`.
The formalism for computing solid Earth tides within the IERS Conventions include the component of deformation induced by the degree 3 terms :cite:p:`Petit:2010tp,Mathews:1997js`.
Catalogs of tide potential, such as ``HW1995`` [see :ref:`tab-catalogs`], can include even higher degree terms, as well as the potentials induced by planetary motion :cite:p:`Hartmann:1995jp`. 

.. plot:: ./background/sphharm34.py
    :show-source-link: False
    :caption: Spherical harmonics for degrees 3 and 4
    :align: center


.. list-table:: Maximum Tide Potential from :cite:t:`Hartmann:1995jp`
    :header-rows: 1
    :align: center

    * - Body
      - Degree
      - V [m\ :sup:`2`/s\ :sup:`2`]
    * - Moon
      - 2
      - 4.41
    * - Moon
      - 3
      - 7.88\ |times|\ 10\ :sup:`-2`
    * - Moon
      - 4
      - 1.41\ |times|\ 10\ :sup:`-3`
    * - Moon
      - 5
      - 2.53\ |times|\ 10\ :sup:`-5`
    * - Moon
      - 6
      - 4.52\ |times|\ 10\ :sup:`-7`
    * - Moon
      - 7
      - 8.06\ |times|\ 10\ :sup:`-9`
    * - Sun
      - 2
      - 1.60
    * - Sun
      - 3
      - 6.80\ |times|\ 10\ :sup:`-5`
    * - Sun
      - 4
      - 2.89\ |times|\ 10\ :sup:`-9`

.. |times|      unicode:: U+00D7 .. MULTIPLICATION SIGN
