.. _release-v3.1.0:

==================
`Release v3.1.0`__
==================

* ``feat``: added more options to compute nutation angles `(#610) <https://github.com/pyTMD/pyTMD/pull/610>`_
* ``feat``: added more options to compute mean obliquity `(#610) <https://github.com/pyTMD/pyTMD/pull/610>`_
* ``feat``: added more options to compute equation of the equinoxes `(#610) <https://github.com/pyTMD/pyTMD/pull/610>`_
* ``refactor``: split equation of the equinoxes into a separate function `(#610) <https://github.com/pyTMD/pyTMD/pull/610>`_
* ``fix``: use ``np.where`` to adjust longitudinal convention `(#610) <https://github.com/pyTMD/pyTMD/pull/610>`_
* ``fix``: table column format types (``i4`` and ``f8``) `(#610) <https://github.com/pyTMD/pyTMD/pull/610>`_
* ``docs``: use ``Lato`` in plots to match RTD sphinx theme `(#610) <https://github.com/pyTMD/pyTMD/pull/610>`_
* ``docs``: use ``rcParams`` to set ``facecolor`` `(#610) <https://github.com/pyTMD/pyTMD/pull/610>`_
* ``test``: add GAST and eqeq test `(#610) <https://github.com/pyTMD/pyTMD/pull/610>`_
* ``test``: add ``URLError`` to exceptions `(#610) <https://github.com/pyTMD/pyTMD/pull/610>`_
* ``docs``: add perigee / perihelion schematics `(#611) <https://github.com/pyTMD/pyTMD/pull/611>`_
* ``docs``: reorganize nodal precession to astronomy `(#611) <https://github.com/pyTMD/pyTMD/pull/611>`_
* ``docs``: add a sample high/low tide plot `(#611) <https://github.com/pyTMD/pyTMD/pull/611>`_
* ``feat``: lazy regional cropping of chunked/netCDF grids via ``tmd.crop`` with dateline-safe longitude windows `(#613) <https://github.com/pyTMD/pyTMD/pull/613>`_
* ``feat``: crop unstructured finite element meshes via element ``isel`` without materializing the full mesh `(#613) <https://github.com/pyTMD/pyTMD/pull/613>`_
* ``fix``: treat 0-360 inclusive grids as global and drop duplicated meridian in wrap crop `(#613) <https://github.com/pyTMD/pyTMD/pull/613>`_
* ``fix``: keep FES amplitudes when phase is 0 (only mask where both amplitude and phase are zero) `(#613) <https://github.com/pyTMD/pyTMD/pull/613>`_
* ``test``: cover regional crop and Pacific longitude wrap, and add FES windowed-read benchmark script `(#613) <https://github.com/pyTMD/pyTMD/pull/613>`_
* ``fix``: reduce constituents before extracting model in ``model.open_dataset()`` `(#616) <https://github.com/pyTMD/pyTMD/pull/616>`_
* ``refactor``: use ``build_logger`` to handle logging to stream and ``logging.getLogger`` in individual functions `(#619) <https://github.com/pyTMD/pyTMD/pull/619>`_
* ``refactor``: use ``np.hypot`` to calculate harmonic amplitudes and euclidean distances `(#619) <https://github.com/pyTMD/pyTMD/pull/619>`_
* ``docs``: add initial finite element section to background and a tip for Love numbers `(#619) <https://github.com/pyTMD/pyTMD/pull/619>`_
* ``docs``: update ``pixi`` tasks for building documentation (removes ``Makefile`` and ``make.bat``) `(#620) <https://github.com/pyTMD/pyTMD/pull/620>`_
* ``refactor``: used ``taplo`` to format ``pyproject.toml`` `(#620) <https://github.com/pyTMD/pyTMD/pull/620>`_
* ``feat``: add function for converting ocean models to SH coefs `(#621) <https://github.com/pyTMD/pyTMD/pull/621>`_
* ``docs``: add more bullets to troubleshooting `(#621) <https://github.com/pyTMD/pyTMD/pull/621>`_
* ``docs``: add more notes to Love/Shida numbers `(#621) <https://github.com/pyTMD/pyTMD/pull/621>`_
* ``docs``: update geoid height figure to add orthometric height `(#621) <https://github.com/pyTMD/pyTMD/pull/621>`_
* ``docs``: add geodetic coordinates figure `(#621) <https://github.com/pyTMD/pyTMD/pull/621>`_
* ``docs``: more blurbs about time `(#621) <https://github.com/pyTMD/pyTMD/pull/621>`_
* ``feat``: add LLN reader to ``earth`` `(#621) <https://github.com/pyTMD/pyTMD/pull/621>`_
* ``feat``: add 3 different LLN datasets `(#621) <https://github.com/pyTMD/pyTMD/pull/621>`_
* ``feat``: added recurrence function for Legendre polynomials `(#621) <https://github.com/pyTMD/pyTMD/pull/621>`_
* ``feat``: added option to calculate fully-normalized Legendre polynomials `(#621) <https://github.com/pyTMD/pyTMD/pull/621>`_
* ``feat``: add spherical harmonic prediction program `(#621) <https://github.com/pyTMD/pyTMD/pull/621>`_
* ``feat``: add minor infer for spherical harmonics `(#621) <https://github.com/pyTMD/pyTMD/pull/621>`_
* ``test``: compare HW95 Legendre versus Mohlenkamp `(#621) <https://github.com/pyTMD/pyTMD/pull/621>`_
* ``fix``: try fetching from both providers `(#621) <https://github.com/pyTMD/pyTMD/pull/621>`_
* ``fix``: put try/except in ``fetch_test_data`` function `(#621) <https://github.com/pyTMD/pyTMD/pull/621>`_
* ``fix``: modulate load Love numbers for FCN resonance `(#622) <https://github.com/pyTMD/pyTMD/pull/622>`_
* ``feat``: add variable attributes to output SH dataset `(#622) <https://github.com/pyTMD/pyTMD/pull/622>`_
* ``test``: add nutation angle check against NOAJ values `(#623) <https://github.com/pyTMD/pyTMD/pull/623>`_
* ``test``: add a sidereal time check against NAOJ values `(#623) <https://github.com/pyTMD/pyTMD/pull/623>`_
* ``fix``: add SH time series functions to ``__all__`` `(#623) <https://github.com/pyTMD/pyTMD/pull/623>`_
* ``feat``: add documentation link checker `(#623) <https://github.com/pyTMD/pyTMD/pull/623>`_
* ``docs``: add citation to Ray and Sanchez 1989 TM `(#623) <https://github.com/pyTMD/pyTMD/pull/623>`_
* ``fix``: mask ATLAS 0 values to ``NaN`` `(#624) <https://github.com/pyTMD/pyTMD/pull/624>`_
* ``fix``: set ``drop`` to ``False`` to prevent problems when merging `(#624) <https://github.com/pyTMD/pyTMD/pull/624>`_
* ``fix``: mask where there are invalid bathymetries `(#625) <https://github.com/pyTMD/pyTMD/pull/625>`_
* ``docs``: update ``ocean_harmonics`` comments and docstrings `(#625) <https://github.com/pyTMD/pyTMD/pull/625>`_
* ``refactor``: move ``ocean_harmonics`` into ``solve`` `(#626) <https://github.com/pyTMD/pyTMD/pull/626>`_
* ``feat``: add initial load tide calculator ``crustal_loading`` `(#626) <https://github.com/pyTMD/pyTMD/pull/626>`_
* ``refactor``: switch convention back to ``clm``/``slm`` `(#627) <https://github.com/pyTMD/pyTMD/pull/627>`_
* ``feat``: added function to estimate seawater densities `(#627) <https://github.com/pyTMD/pyTMD/pull/627>`_
* ``fix``: node equilibrium for  `#628 <https://github.com/pyTMD/pyTMD/issues/628>`_ `(#629) <https://github.com/pyTMD/pyTMD/pull/629>`_
* ``fix``: create download directory for test data if calling ``_zenodo`` or ``_figshare`` functions directly `(#629) <https://github.com/pyTMD/pyTMD/pull/629>`_
* ``feat``: add back USAP-DC access program `(#630) <https://github.com/pyTMD/pyTMD/pull/630>`_
* ``docs``: add USAP-DC registration information for getting access tokens `(#630) <https://github.com/pyTMD/pyTMD/pull/630>`_
* ``feat``: add Green's function crustal loading calculator `(#631) <https://github.com/pyTMD/pyTMD/pull/631>`_
* ``refactor``: split ``SHA`` from crustal loading `(#631) <https://github.com/pyTMD/pyTMD/pull/631>`_
* ``feat``: added grid cell area calculators for different crs `(#631) <https://github.com/pyTMD/pyTMD/pull/631>`_
* ``feat``: added function to convert coordinates to geographic `(#631) <https://github.com/pyTMD/pyTMD/pull/631>`_
* ``feat``: add ``io`` wrapper functions for datasets and datatrees `(#631) <https://github.com/pyTMD/pyTMD/pull/631>`_
* ``feat``: add more stable factorial terms to prevent overflow `(#631) <https://github.com/pyTMD/pyTMD/pull/631>`_
* ``feat``: add new ``EarthModel`` and ``EarthLayer`` classes `(#631) <https://github.com/pyTMD/pyTMD/pull/631>`_
* ``docs``: add harmonic summation and greens function to API refs `(#631) <https://github.com/pyTMD/pyTMD/pull/631>`_
* ``docs``: add Green's function section `(#631) <https://github.com/pyTMD/pyTMD/pull/631>`_
* ``refactor``: set default mean longitudes to use ``IERS`` `(#631) <https://github.com/pyTMD/pyTMD/pull/631>`_
* ``docs``: forgot to add Hobson (1955) to the bibtex file
* ``refactor``: use a unit vector relation for P21 `(#634) <https://github.com/pyTMD/pyTMD/pull/634>`_
* ``refactor``: use 100 reference angles for cached Green functions `(#634) <https://github.com/pyTMD/pyTMD/pull/634>`_
* ``docs``: add asymptotic solution used in horizontal displacements `(#634) <https://github.com/pyTMD/pyTMD/pull/634>`_
* ``refactor``: use half spherical cap radius for coincident points `(#634) <https://github.com/pyTMD/pyTMD/pull/634>`_
* ``docs``: expand documentation for hyperslab functions `(#635) <https://github.com/pyTMD/pyTMD/pull/635>`_
* ``feat``: add ``greater_close`` and ``less_close`` comparison functions to ``math`` `(#635) <https://github.com/pyTMD/pyTMD/pull/635>`_
* ``refactor``: rename some variables for continuity `(#635) <https://github.com/pyTMD/pyTMD/pull/635>`_

.. __: https://github.com/pyTMD/pyTMD/releases/tag/3.1.0
