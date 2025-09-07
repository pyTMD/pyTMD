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

date: 06 August 2025
bibliography: pytmd-refs.bib
---

# Summary

`pyTMD` is an open-source tidal prediction software that aims to simplify the calculation of ocean and Earth tides. It is designed to be able to handle a wide range of model formats and can incorporate different physics schemes. This flexibility allows `pyTMD` to be tailored to fit specific use cases, while still allowing ease of use.

# Statement of need

There are several ocean tide prediction software options available. The OSU Tidal Inversion Software (`OTIS`) and OSU Tidal Prediction Software (`OTPS`) were developed by Lana Erofeeva and Gary Egbert. The Tidal Model Driver ([`TMD`](https://github.com/EarthAndSpaceResearch/TMD_Matlab_Toolbox_v2.5)) MATLAB Toolbox was developed by Laurie Padman, Lana Erofeeva and Susan Howard. An updated version of the MATLAB Toolbox ([`TMD3`](https://github.com/chadagreene/tide-model-driver)) was developed by Chad Greene. The NASA GSFC PREdict Tidal Heights (`PERTH3`) software was developed by Richard Ray and Remko Scharroo. An updated and more versatile version of the NASA GSFC Fortran software ([`PERTH5`](https://codeberg.org/rray/perth5)) was developed by Richard Ray. [`pyFES`](https://cnes.github.io/aviso-fes/) was produced by LEGOS, NOVELTIS and CLS Ocean and Climate Division, and funded by CNES [@Lyard:2025tr]. These software options are typically created by or for the model providers, and support their specific model formats.

`pyTMD` is a generalized tide program that allows users to calculate both tide deflections and currents. As a tide model driver, `pyTMD` can read from a broad suite of models, and use different physics schemes in the internal calculations. Over 50 different models are presently supported within `pyTMD`, and additional models can be defined with a JSON file. 

`pyTMD` was designed to be used by beginners and scientific researchers alike. The online documentation contains background information for both tidal modeling and prediction. The software has been used in a number of scientific publications for modeling regional tides [@Freer:2023bt; @Millan:2023ju; @Sutterley:2019cl], modeling global tides [@Gregg:2024ky; @Paprotny:2024et], and creating several Earth observation datasets [@ENVEO:2021jh; @Smith:2024fd; @Smith:2024cv]. It has also been leveraged within larger earth-observation software packages [@Fitzpatrick:2024iv; @BishopTaylor:2025dc]. 

# Functionality

## Ocean and Load Tides

Tides are frequently decomposed into harmonic constants, or constituents, associated with the relative positions of the sun, moon and Earth [@Padman:2018cv]. For ocean and load tides, `pyTMD.io` contains routines for reading major constituent values (amplitude and phase) from commonly available tide models, which typically fall within a few general formats: `OTIS-binary` [@Egbert:2002ge; @Padman:2008ec], `OTIS-compact`, `OTIS-netcdf`, `TMD3-netcdf` [@Greene:2024di], `GOT-ascii` [@Ray:1999vm], `GOT-netcdf`, `FES-ascii` [@LeProvost:1994ie] and `FES-netcdf` [@Lyard:2025tr]. Information for each of the supported tide models is stored within a JSON database. `pyTMD.io` reads or computes the amplitude and phase lag of tide model constituents, which are then interpolated to the output spatial coordinates. 

`pyTMD` uses the astronomical argument formalism outlined in @Doodson:1921kt to predict ocean and load tides. After the computation of the amplitude and phase lag for the spatial coordinates, `pyTMD` computes the temporal elements using by 1) calculating the astronomical angles ($S$, $H$, $P$, $N$, $P_s$) [@Meeus:1991vh; @Simon:1994vo], 2) combining these angles with the "Doodson numbers" in a Fourier series to compute the equilibrium tide phase ($G$) of each constituent, and 3) computing the 18.6-year nodal amplitude and phase corrections ($f$ and $u$) of each constituent [@Doodson:1921kt; @Dietrich:1980ua; @Pugh:2014di]. The spatial and temporal components are then combined, and the output tidal time series is calculated through a summation over all constituents [@Egbert:2002ge]. The contributions of "minor" constituents can then be estimated using inference methods [@Schureman:1958ty; @Ray:2017jx].

Long-period ocean tides can additionally be predicted assuming an "equilibrium response" [@Doodson:1921kt; @Cartwright:1971iz; @Cartwright:1973em]. Here, the oceanic surface is estimated to respond instantaneously to the tide-producing forces of the moon and sun, and is not influenced by inertia, currents or the irregular distribution of land [@Schureman:1958ty; @Proudman:1960jj; @Ray:2014fu]. 

Separately, `pyTMD` has some baseline capability to decompose a time series into tidal constants (constituent amplitudes and phases) using the harmonic method outlined in @Foreman:1989dt and @Foreman:2009bg.

## Pole Tides

The Earth's rotation axis is inclined at an angle of 23.5 degrees to the celestial pole, of which it rotates about every 26,000 years [@Kantha:2000vo]. Superimposed on this long-term precession, the rotation axis shifts with respect to its mean pole position due to nutation, Chandler wobble, annual variations, and other processes [@Desai:2002ev; @Wahr:1985gr]. Load and ocean pole tides are driven by variations in the Earth's rotation axis with respect to its mean position, along with corresponding elastic responses and secondary effects [@Desai:2002ev; @Desai:2015jr; @Wahr:1985gr]. `pyTMD.predict` estimates pole tide variations following IERS Conventions by differencing the daily IERS polar motion "finals" from the reference "secular" pole positions [@Petit:2010tp]. 

## Solid Earth Tides

The tidal deformation of the solid Earth is modeled in `pyTMD` using the Love and Shida number formalism from @Wahr:1981ea and @Mathews:1997js as described in @Petit:2010tp. `pyTMD.astro` has options for calculating approximate ephemerides following @Meeus:1991vh and @Montenbruck:1989uk or using high-resolution JPL ephemerides from @Park:2021fa using the `jplephem` package [@Rhodes:2011to]. The calculation for solid Earth tides includes the frequency-dependent and dissipative mantle anelasticity corrections from @Mathews:1997js. 

## Time

`pyTMD` uses the `timescale` library [@Sutterley:2025cx] to manage temporal conversions, calculate "dynamical" time scales, and calculate Earth Orientation Parameters (EOPs). 

# Acknowledgements

`pyTMD` was first supported through an appointment to the NASA Postdoctoral Program (NPP) at NASA Goddard Space Flight Center (GSFC), and currently supported under the NASA Cryospheric Sciences Program (Grant Number 80NSSC22K0379). The software was initially developed with the goal of supporting science applications for airborne and satellite altimetry in preparation for the launch of the NASA ICESat-2 mission. It was designed for scientific and technical purposes, and not for coastal navigation or applications risking life or property.

We wish to acknowledge the invaluable comments, contributions, and support from Karen Alley (University of Manitoba), Robbi Bishop-Taylor (Geoscience Australia), Kelly Brunt (NSF) and Richard Ray (NASA GSFC) towards the development of `pyTMD`. We additionally wish to acknowledge the comments, issues and discussions of all contributors to the `pyTMD` GitHub repository.

# References
