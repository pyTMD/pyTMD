:orphan:

.. _fig-sphharm:

Spherical Harmonics
-------------------

The tide potential at colatitude :math:`\theta` and longitude :math:`\phi` is often expressed in terms of spherical harmonic functions :math:`Y_l^m(\theta,\phi)` of degree :math:`l` and order :math:`m` :cite:p:`Doodson:1921kt`.
The degree 2 spherical harmonic terms are the dominant source of tidal excitation, and induce the semi-diurnal (from :math:`Y_2^2`), diurnal (from :math:`Y_2^1`) and long-period (from :math:`Y_2^0`) tides.

.. plot:: ./background/sphharm.py
    :show-source-link: False
    :caption: Spherical harmonics for degree 2
    :align: center

Asymmetry in the tide potential can lead to a dependence on higher degree harmonics, most notably the degree 3 and 4 terms.
These terms are smaller compared to those of degree 2, but have been detected at both local and global scales :cite:p:`Ray:2020gn`.
For example, GNSS stations and superconducting gravimeters can be sensitive enough to be affected by these higher degree terms :cite:p:`Hartmann:1995jp,Petit:2010tp`.
IERS standards for computing solid Earth tides include the deformation induced by the degree 3 terms :cite:p:`Petit:2010tp`.
Tidal catalogs can include even higher degree terms, as well as the tide potentials induced by planetary motion :cite:p:`Hartmann:1995jp`. 

.. plot:: ./background/sphharm34.py
    :show-source-link: False
    :caption: Spherical harmonics for degrees 3 and 4
    :align: center

