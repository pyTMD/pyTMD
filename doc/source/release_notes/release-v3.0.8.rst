.. _release-v3.0.8:

##################
`Release v3.0.8`__
##################

* ``feat``: added inverse distance weighting (IDW) extrapolation `(#577) <https://github.com/pyTMD/pyTMD/pull/577>`_
* ``refactor``: move ellipsoid and love number parameters to ``earth`` module `(#577) <https://github.com/pyTMD/pyTMD/pull/577>`_
* ``refactor``: use ``scipy.spatial.KDTree`` as ``cKDTree`` is now an alias `(#577) <https://github.com/pyTMD/pyTMD/pull/577>`_
* ``feat``: add ``exists`` to ``URL`` class `(#577) <https://github.com/pyTMD/pyTMD/pull/577>`_
* ``refactor``: improve ``body_tide`` computational times `(#578) <https://github.com/pyTMD/pyTMD/pull/578>`_
* ``test``: add unit tests for body tides using HW1995 and W1990 `(#578) <https://github.com/pyTMD/pyTMD/pull/578>`_
* ``test``: update random integers for new ``numpy`` syntax `(#578) <https://github.com/pyTMD/pyTMD/pull/578>`_
* ``fix``: aberration in solar longitude had radians conversion twice `(#578) <https://github.com/pyTMD/pyTMD/pull/578>`_
* ``fix``: ``solar_distance`` needed ``ephemerides`` keyword argument `(#578) <https://github.com/pyTMD/pyTMD/pull/578>`_
* ``docs``: update references to IERS chapter 5 tables `(#578) <https://github.com/pyTMD/pyTMD/pull/578>`_
* ``chore``: update ``pixi.lock`` to version 7 `(#578) <https://github.com/pyTMD/pyTMD/pull/578>`_
* ``docs``: add more information to getting started section `(#579) <https://github.com/pyTMD/pyTMD/pull/579>`_
* ``feat``: updated constituent parameters function to use dictionaries `(#580) <https://github.com/pyTMD/pyTMD/pull/580>`_
* ``feat``: add Kronecker delta function `(#580) <https://github.com/pyTMD/pyTMD/pull/580>`_
* ``docs``: add zenith angle function and term to glossary `(#580) <https://github.com/pyTMD/pyTMD/pull/580>`_
* ``docs``: use lower case terms where applicable `(#580) <https://github.com/pyTMD/pyTMD/pull/580>`_
* ``docs``: expand spherical harmonics page `(#580) <https://github.com/pyTMD/pyTMD/pull/580>`_
* ``docs``: use ``LaTeX`` math in docstrings `(#580) <https://github.com/pyTMD/pyTMD/pull/580>`_
* ``test``: update constituent and math tests `(#580) <https://github.com/pyTMD/pyTMD/pull/580>`_
* ``chore``: bump ``pixi`` ci versions `(#580) <https://github.com/pyTMD/pyTMD/pull/580>`_
* ``ci``: add ``sphinx`` build check `(#580) <https://github.com/pyTMD/pyTMD/pull/580>`_
* ``test``: add ``ValueError`` to NOAA tests `(#580) <https://github.com/pyTMD/pyTMD/pull/580>`_
* ``docs``: add to ``Astronomy`` `(#581) <https://github.com/pyTMD/pyTMD/pull/581>`_
* ``docs``: define ``n`` in the legendre polynomial calculation `(#581) <https://github.com/pyTMD/pyTMD/pull/581>`_
* ``docs``: bold font in some lists `(#581) <https://github.com/pyTMD/pyTMD/pull/581>`_
* ``ci``: archive sphinx warnings to artifact `(#582) <https://github.com/pyTMD/pyTMD/pull/582>`_
* ``docs``: add ``numfig_format`` filter `(#583) <https://github.com/pyTMD/pyTMD/pull/583>`_
* ``ci``: add ``--fail-on-warning`` and ``--keep-going`` to ``SPHINXOPTS`` `(#583) <https://github.com/pyTMD/pyTMD/pull/583>`_
* ``docs``: expand ephemerides section `(#584) <https://github.com/pyTMD/pyTMD/pull/584>`_
* ``docs``: eq-eq had duplicate info from sidereal time `(#584) <https://github.com/pyTMD/pyTMD/pull/584>`_
* ``docs``: add approximate method table `(#584) <https://github.com/pyTMD/pyTMD/pull/584>`_
* ``docs``: add tide correction table `(#584) <https://github.com/pyTMD/pyTMD/pull/584>`_
* ``docs``: add to glossary `(#585) <https://github.com/pyTMD/pyTMD/pull/585>`_
* ``docs``: add extrapolation notebook `(#585) <https://github.com/pyTMD/pyTMD/pull/585>`_
* ``feat``: add ``is_geographic`` to ``inpaint`` `(#585) <https://github.com/pyTMD/pyTMD/pull/585>`_
* ``refactor``: use ``_build_tree`` in ``inpaint`` `(#585) <https://github.com/pyTMD/pyTMD/pull/585>`_
* ``fix``: ``lint`` issues by breaking up imports `(#586) <https://github.com/pyTMD/pyTMD/pull/586>`_
* ``fix``: ``lint`` apply ``E401`` fixes (break up imports) `(#586) <https://github.com/pyTMD/pyTMD/pull/586>`_
* ``fix``: ``ruff check --select SIM118,UP034,F541,UP004 --fix .`` `(#586) <https://github.com/pyTMD/pyTMD/pull/586>`_
* ``docs``: add steps for calculating solid earth tides `(#587) <https://github.com/pyTMD/pyTMD/pull/587>`_
* ``docs``: add initial gravity tide section `(#587) <https://github.com/pyTMD/pyTMD/pull/587>`_
* ``docs``: renumber all equations after gravity tides addition `(#587) <https://github.com/pyTMD/pyTMD/pull/587>`_
* ``docs``: convert Install page to ``myst-nb`` `(#588) <https://github.com/pyTMD/pyTMD/pull/588>`_
* ``feat``: allow for encrypted ftp connections `(#589) <https://github.com/pyTMD/pyTMD/pull/589>`_
* ``feat``: can get github url of an item in the project repo `(#589) <https://github.com/pyTMD/pyTMD/pull/589>`_
* ``docs``: update resources list `(#589) <https://github.com/pyTMD/pyTMD/pull/589>`_
* ``docs``: add some notes to background `(#589) <https://github.com/pyTMD/pyTMD/pull/589>`_
* ``docs``: add Lambeck (1980) to citations `(#589) <https://github.com/pyTMD/pyTMD/pull/589>`_

.. __: https://github.com/pyTMD/pyTMD/releases/tag/3.0.8
