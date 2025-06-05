---
title: 'pyTMD: Python-based tidal prediction software'
tags:
  - geophysics
  - oceanography
  - geodesy
  - tides

authors:
  - name: Tyler C. Sutterley
    orcid: 0000-0002-6964-1194
    corresponding: true
    affiliation: 1
  - name: Susan L. Howard
    orcid: 0000-0002-9183-0178
    equal-contrib: true
    affiliation: 2
  - name: Laurie Padman
    orcid: 0000-0003-2010-642X
    equal-contrib: true
    affiliation: 2
  - name: Matthew R. Siegfried
    orcid: 0000-0002-0868-4633
    equal-contrib: true
    affiliation: 3

affiliations:
 - name: Polar Science Center, Applied Physics Laboratory, University of Washington, United States
   index: 1
 - name: Earth & Space Research, United States
   index: 2
 - name: Department of Geophysics, Colorado School of Mines, United States
   index: 3

date: 05 June 2024
bibliography: pytmd-refs.bib
---

# Summary

`pyTMD` is an open-source tidal prediction software that aims to simplify the calculation of ocean and earth tides. It is designed to be able to handle a wide range of model formats and can incorporate different physics schemes. This flexibility allows `pyTMD` to be tailored to fit specific use cases, while still allowing ease of use.

# Statement of need

`pyTMD` is a generalized tide program that allows users to calculate both tide deflections and currents. As a tide model driver, `pyTMD` can read from a broad suite of models, and use different physics schemes in the internal calculations. Over 50 different models are presently supported within `pyTMD`, and additional models can be used when defined with a JSON file. `pyTMD` also supports the calculation of other sources of vertical and horizontal variability, such as from solid Earth and pole tides. `pyTMD` includes documentation and background information for tide modeling, which includes multiple `Jupyter` notebooks that demonstrate core functionality.

`pyTMD` was designed to be used by scientific researchers and beginners alike. It has been used in a number of scientific publications [@Freer:2023bt; @Millan:2023ju; @Sutterley:2019cl], and for the creation of several Earth observation datasets [@Smith:2024fd; @Smith:2024cv]. 

# Functionality

## Time

`pyTMD` uses the `timescale` library [@Sutterley:2025cx] to manage temporal conversions, calculate "dynamical" time scales, and calculate Earth Orientation Parameters (EOPs). 

## Ocean and Load Tides

Tides are frequently decomposed into harmonic constants, or constituents, associated with the relative positions of the sun, moon and Earth. For ocean and load tides, `pyTMD.io` contains routines for reading major constituent values from commonly available tide models, which typically fall within a few general formats: `OTIS-binary` [@Egbert:2002ge; @Padman:2008ec], `OTIS-compact`, `OTIS-netcdf`, `TMD3-netcdf` [@Greene:2024di], `GOT-ascii` [@Ray:1999vm], `GOT-netcdf`, `FES-ascii` [@LeProvost:1994ie] and `FES-netcdf`. These models can be global or regional, and typically fit one of following categories: 1) empirically adjusted models, 2) barotropic hydrodynamic models constrained by data assimilation, and 3) unconstrained hydrodynamic models [@Stammer:2014ci]. Information for each of the supported tide models is stored within a JSON database that is supplied with `pyTMD`. `pyTMD.io` reads or computes the amplitude and phase lag of tide model constituents, which are then interpolated to the output spatial coordinates. 

`pyTMD` uses the astronomical argument formalism outlined in @Doodson:1921kt for the prediction of ocean and load tides. After the computation of the amplitude and phase lag for the spatial coordinates, `pyTMD` computes the temporal elements using by 1) calculating the astronomical angles ($S$, $H$, $P$, $N$, $P_s$) with `pyTMD.astro` [@Meeus:1991vh; @Simon:1994vo], 2) combining these angles with the "Doodson numbers" in a Fourier series to compute the equilibrium tide phase ($G$) of each constituent with `pyTMD.arguments`, and 3) computing the 18.6-year nodal amplitude and phase corrections ($f$ and $u$) of each constituent also with `pyTMD.arguments` [@Doodson:1921kt; @Dietrich:1980ua; @Pugh:2014di]. Finally, `pyTMD.predict` combines the spatial and temporal components, and takes the sum over all constituents to calculate the tidal time series [@Egbert:2002ge]. `pyTMD.predict` can additionally estimate the contributions of "minor" constituents using inferrence methods to calculate more of the tidal spectrum [@Schureman:1958ty; @Ray:2017jx]. `pyTMD.compute` contains some higher-level functionality to wrap the prediction functions for more convenient access.

Separately, `pyTMD.compute` can predict long-period ocean tides assuming an "equilibrium response" [@Doodson:1921kt]. Here, the oceanic surface instantaneously responds to the tide-producing forces of the moon and sun, and is not influenced by inertia, currents or the irregular distribution of land [@Schureman:1958ty]. While the equilibrium condition is rarely satisfied for shorter period tides, long period tides can be well approximated as these simplified responses [@Proudman:1960jj; @Ray:2014fu]. `pyTMD.predict` uses the tide potential values from @Cartwright:1971iz and @Cartwright:1973em to calculate the tide deflections.

`pyTMD` software has some baseline solution capability for tidal constants using the harmonic method of decomposition [@Foreman:1989dt; @Foreman:2009bg]. `pyTMD` takes into account the 18.6-year nodal corrections within the design matrix for the least-squares solver [@Egbert:2002ge; @Foreman:2009bg].

$$
h(t) = z_0 +\sum_{k=1}^{n} f_k(t) \left[X_k\cos{\left(G_k(t)+u_k(t)\right)} - Y_k\sin{\left(G_k(t)+u_k(t)\right)}\right] + \sigma
$$

where $h(t)$ is the sea level height at time $t$, $z_0$ is mean sea level, $f_k(t)$ is the (nodal) amplitude modulation in constituent $k$, $G_k(t)$ is the phase of the equilibrium tide, $u_k(t)$ is the (nodal) phase correction angle, and $\sigma$ is the uncertainty. The amplitude $A_k$ and phase $\theta_k$ of the constituent are calculated using the solution coefficients $X_k$ and $Y_k$.

$$
A_k = \sqrt{X_k^2 + Y_k^2}
$$
$$
\theta_k = \tan^{-1}{\left(\frac{-Y_k}{X_k}\right)}
$$

## Pole Tides

Load and ocean pole tides are driven by variations in the Earth's rotation axis with respect to its mean position, along with corresponding responses and secondary effects [@Desai:2002ev; @Desai:2015jr; @Wahr:1985gr]. `pyTMD.predict` calculates the polar motion values using the "finals" file provided by IERS, and the reference "secular" pole position following the latest IERS conventions [@Petit:2010tp]. `pyTMD.compute` contains a wrapper function for computing the radial components of the pole tide deflections.

## Solid Earth Tides

The tidal deformation of the Solid Earth is modeled using the Love and Shida number formalism described in [@Wahr:1981ea; @Mathews:1997js] and formalized in the IERS Conventions [@Petit:2010tp]. The tide potential inducing these deformations is a function of the position of the sun and moon with respect to the Earth. `pyTMD.astro` has options for calculating approximate ephemerides following @Meeus:1991vh and @Montenbruck:1989uk or using JPL ephemerides from @Park:2021fa using the `jplephem` package [@Rhodes:2011to]. `pyTMD.predict` computes Solid Earth tides including frequency-dependent corrections and dissipative mantle anelasticity corrections following @Mathews:1997js. `pyTMD.compute` contains a wrapper function for computing the solar and lunar ephemerides for a given date, and estimating the radial components of the solid Earth tide deflections.

# Acknowledgements

`pyTMD` was first supported through an appointment to the NASA Postdoctoral Program (NPP) at NASA Goddard Space Flight Center (GSFC), and currently supported under the NASA Cryospheric Sciences Program (Grant Number 80NSSC22K0379). The software was initially developed with the goal of supporting science applications for airborne and satellite altimetry in preparation for the launch of the NASA ICESat-2 mission. It was designed for scientific and technical purposes, and not for coastal navigation or applications risking life or property.

We acknowledge invaluable comments, contributions, and support from Karen Alley (University of Manitoba), Robbi Bishop-Taylor (Geoscience Australia), Kelly Brunt (NSF) and Richard Ray (NASA GSFC) towards the development of `pyTMD`, in addition to the comments, issues and discussions of all contributors to the GitHub repository.

The Tidal Model Driver ([`TMD`](https://github.com/EarthAndSpaceResearch/TMD_Matlab_Toolbox_v2.5)) Matlab Toolbox was developed by Laurie Padman, Lana Erofeeva and Susan Howard. An updated version of the TMD Matlab Toolbox ([`TMD3`](https://github.com/chadagreene/tide-model-driver)) was developed by Chad Greene. The OSU Tidal Inversion Software (OTIS) and OSU Tidal Prediction Software (OTPS) were developed by Lana Erofeeva and Gary Egbert (copyright OSU, licensed for non-commercial use). The NASA Goddard Space Flight Center (GSFC) PREdict Tidal Heights (PERTH3) software was developed by Richard Ray and Remko Scharroo. An updated and more versatile version of the NASA GSFC tidal prediction software ([`PERTH5`](https://codeberg.org/rray/perth5)) was developed by Richard Ray.

# References
