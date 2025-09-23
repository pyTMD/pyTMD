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

date: 23 September 2025
bibliography: pytmd-refs.bib
---

# Summary

`pyTMD` is an open-source tidal prediction software that aims to simplify the calculation of ocean and Earth tides. It is not an ocean or load tide model, but a tool for using tidal constants provided by tide models to predict the height deflections or currents at particular locations and times. It is designed to be able to handle a wide range of model formats and can incorporate different physics schemes. This flexibility allows `pyTMD` to be tailored to fit specific use cases, while still allowing ease of use.

# Statement of need

There are several ocean tide prediction software options available. The OSU Tidal Inversion Software (`OTIS`) and OSU Tidal Prediction Software (`OTPS`) are Fortran programs developed by Lana Erofeeva and Gary Egbert for the [TPXO](https://www.tpxo.net/home) family of models. The Tidal Model Driver ([`TMD`](https://github.com/EarthAndSpaceResearch/TMD_Matlab_Toolbox_v2.5)) is a MATLAB Toolbox developed by Laurie Padman, Lana Erofeeva and Susan Howard for the same family of models. An updated version of the MATLAB Toolbox ([`TMD3`](https://github.com/chadagreene/tide-model-driver)) was developed by Chad Greene. [`pyFES`](https://cnes.github.io/aviso-fes/) is a Python library produced by LEGOS, NOVELTIS and CLS Ocean and Climate Division, and funded by CNES [@Lyard:2025tr] for the Finite Element Solution (FES) family of models. The NASA GSFC PREdict Tidal Heights (`PERTH3`) software is a Fortran program developed by Richard Ray and Remko Scharroo for the Goddard Ocean Tide (GOT) family of models. An updated and more versatile version of the NASA GSFC Fortran software ([`PERTH5`](https://codeberg.org/rray/perth5)) was developed by Richard Ray. `PERTH5` is a more generalized program, and can read from multiple different tide model formats. These software options are typically created by or for the model providers, and, with the exception of `PERTH5`, singly support their specific model formats.  

`pyTMD` is a generalized tide program that allows users to calculate both tide deflections and currents from a broad suite of models. Over 50 different models are presently supported within `pyTMD`, and additional model schemas can be defined with a JSON file. 

`pyTMD` was designed to be used by beginners and scientific researchers alike. The online documentation contains background information for both tidal modeling and prediction. The software has been used in a number of scientific publications for modeling regional tides [@Freer:2023bt; @Millan:2023ju; @Sutterley:2019cl], modeling global tides [@Gregg:2024ky; @Paprotny:2024et], and creating several Earth observation datasets [@ENVEO:2021jh; @Smith:2024fd; @Smith:2024cv]. It has also been leveraged within larger earth-observation software packages [@Fitzpatrick:2024iv; @BishopTaylor:2025dc]. 

# Functionality

## Ocean and Load Tides

With the harmonic method, tides are decomposed into harmonic constants, or constituents, associated with the relative positions of the sun, moon and Earth [@Padman:2018cv]. These constituents are typically classified into different "species" based on their approximate period: short-period, semi-diurnal, diurnal, and long-period. `pyTMD.io` contains routines for reading major constituent values (amplitude and phase) from commonly available tide models, which typically fall within a few general formats: `OTIS-binary` [@Egbert:2002ge; @Padman:2008ec], `OTIS-compact`, `OTIS-netcdf`, `TMD3-netcdf` [@Greene:2024di], `GOT-ascii` [@Ray:1999vm], `GOT-netcdf`, `FES-ascii` [@LeProvost:1994ie] and `FES-netcdf` [@Lyard:2025tr]. Information for each of the supported tide models is stored within a JSON database. For tidal predictions, `pyTMD.io` interpolates the amplitude and phase lag of tide model constituents to sets of spatial coordinates. 

`pyTMD` uses the astronomical argument formalism outlined in @Doodson:1921kt to compute the temporal elements. Temporal conversions and "dynamical" time scales are managed in `pyTMD` with the `timescale` library [@Sutterley:2025cx]. For a set of temporal values, `pyTMD` 1) calculates the astronomical angles ($S$, $H$, $P$, $N$, $P_s$) [@Meeus:1991vh; @Simon:1994vo], 2) combines these angles with the "Doodson numbers" in a Fourier series to compute each constituent's equilibrium tide phase ($G$), and 3) computes each constituent's 18.6-year nodal amplitude and phase corrections ($f$ and $u$) [@Doodson:1921kt; @Dietrich:1980ua; @Pugh:2014di]. The spatial and temporal components are then combined, and the output tidal time series is calculated through a summation over all constituents [@Egbert:2002ge]. Additional "minor" constituents can be "inferred" to include more of the tidal spectrum [@Schureman:1958ty; @Ray:2017jx].

Long-period ocean tides can independently be predicted assuming an "equilibrium response" [@Doodson:1921kt; @Cartwright:1971iz; @Cartwright:1973em]. Here, the oceanic surface is estimated to respond instantaneously to the tide-producing forces of the moon and sun, and is not influenced by inertia, currents or the irregular distribution of land [@Schureman:1958ty; @Proudman:1960jj; @Ray:2014fu].

## Pole Tides

The Earth's rotation axis is inclined at an angle of 23.5 degrees to the celestial pole, which it rotates about every 26,000 years [@Kantha:2000vo]. Superimposed on this long-term precession, the rotation axis shifts due to nutation, Chandler wobble, annual variations, and other processes [@Desai:2002ev; @Wahr:1985gr]. Load and ocean pole tides are driven by these variations in the Earth's rotation axis, along with corresponding elastic responses and secondary effects [@Desai:2002ev; @Desai:2015jr; @Wahr:1985gr]. `pyTMD` follows IERS Conventions [@Petit:2010tp] to estimate load and ocean pole tide variations, which are based on @Desai:2002ev. The daily IERS polar motion "finals" are kept up-to-date using the `timescale` library [@Sutterley:2025cx]. 

## Solid Earth Tides

The tidal deformation of the solid Earth can be modeled in `pyTMD` using one of the following two methods: 1) the ephemerides formalism from @Wahr:1981ea and @Mathews:1997js as described in @Petit:2010tp, and 2) the tide catalog formalism outlined in @Cartwright:1971iz. For the ephemerides method, `pyTMD.astro` has options for calculating approximate ephemerides following @Meeus:1991vh and @Montenbruck:1989uk or using high-resolution JPL ephemerides from @Park:2021fa with the `jplephem` package [@Rhodes:2011to]. For both cases, the calculation for solid Earth tides includes multiple Love and Shida number corrections including the frequency-dependent and dissipative mantle anelasticity corrections from @Mathews:1997js. 

# Acknowledgements

`pyTMD` was first supported through an appointment to the NASA Postdoctoral Program (NPP) at NASA Goddard Space Flight Center (GSFC), and currently supported under the NASA Cryospheric Sciences Program (Grant Number 80NSSC22K0379). The software was developed with the goal of supporting science applications for airborne and satellite altimetry in preparation for the launch of the NASA ICESat-2 mission. It was designed for scientific and technical purposes, and not for coastal navigation or applications risking life or property.

We wish to acknowledge the invaluable comments, contributions, and support from Karen Alley (University of Manitoba), Robbi Bishop-Taylor (Geoscience Australia), Kelly Brunt (NSF) and Richard Ray (NASA GSFC) towards the development of `pyTMD`. We additionally wish to acknowledge the comments, issues and discussions of all contributors to the `pyTMD` GitHub repository.

# References
