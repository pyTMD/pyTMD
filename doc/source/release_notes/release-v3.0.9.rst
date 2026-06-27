.. _release-v3.0.9:

==================
`Release v3.0.9`__
==================

* ``docs``: add "edit on github" link `(#591) <https://github.com/pyTMD/pyTMD/pull/591>`_
* ``fix``: resolve constituent file paths with ``from_dict`` for  `#417 <https://github.com/pyTMD/pyTMD/issues/#417>`_ `(#592) <https://github.com/pyTMD/pyTMD/pull/592>`_
* ``test``: added a regression test asserting ``from_dict`` and ``from_file`` produce equivalent models both with and without a model directory. `(#592) <https://github.com/pyTMD/pyTMD/pull/592>`_
* ``feat``: added lunisolar equatorial coordinate functions `(#593) <https://github.com/pyTMD/pyTMD/pull/593>`_
* ``docs``: added constituent notations `(#593) <https://github.com/pyTMD/pyTMD/pull/593>`_
* ``docs``: added ecliptic plot `(#593) <https://github.com/pyTMD/pyTMD/pull/593>`_
* ``docs``: added zenith angle plot `(#593) <https://github.com/pyTMD/pyTMD/pull/593>`_
* ``feat``: added spherical linear interpolation (slerp) `(#593) <https://github.com/pyTMD/pyTMD/pull/593>`_
* ``docs``: added reference for slerp (Shoemake 1985) `(#593) <https://github.com/pyTMD/pyTMD/pull/593>`_
* ``docs``: square brackets for internal references `(#593) <https://github.com/pyTMD/pyTMD/pull/593>`_
* ``refactor``: replace assertions with exceptions `(#593) <https://github.com/pyTMD/pyTMD/pull/593>`_
* ``docs``: added geocentric Cartesian coordinate plot `(#594) <https://github.com/pyTMD/pyTMD/pull/594>`_
* ``docs``: fill in quadrants in cartesian plot `(#595) <https://github.com/pyTMD/pyTMD/pull/595>`_
* ``docs``: add link to doodson number in constituent table `(#595) <https://github.com/pyTMD/pyTMD/pull/595>`_
* ``docs``: add table of tidal classifications `(#595) <https://github.com/pyTMD/pyTMD/pull/595>`_
* ``docs``: add precession-nutation figure `(#596) <https://github.com/pyTMD/pyTMD/pull/596>`_
* ``docs``: set figure facecolor to match rtd theme `(#596) <https://github.com/pyTMD/pyTMD/pull/596>`_
* ``docs``: set figures without axes facecolor to match rtd theme `(#596) <https://github.com/pyTMD/pyTMD/pull/596>`_
* ``docs``: add a harmonic analysis section to background `(#597) <https://github.com/pyTMD/pyTMD/pull/597>`_
* ``docs``: bump all reference numbers after new section `(#597) <https://github.com/pyTMD/pyTMD/pull/597>`_
* ``docs``: add ``plot_html_show_source_link`` to ``conf.py`` `(#597) <https://github.com/pyTMD/pyTMD/pull/597>`_
* ``docs``: specify ``plot_formats`` for ``matplotlib`` ``plot_directive`` `(#597) <https://github.com/pyTMD/pyTMD/pull/597>`_
* ``docs``: spherical harmonics to background toc `(#598) <https://github.com/pyTMD/pyTMD/pull/598>`_
* ``docs``: number spherical harmonic equations `(#598) <https://github.com/pyTMD/pyTMD/pull/598>`_
* ``refactor``: standardize use of lambda (``lmda``) to denote longitudes `(#598) <https://github.com/pyTMD/pyTMD/pull/598>`_
* ``docs``: add section on ``type`` variable usage `(#599) <https://github.com/pyTMD/pyTMD/pull/599>`_
* ``docs``: add section on detecting high and low tides `(#599) <https://github.com/pyTMD/pyTMD/pull/599>`_
* ``refactor``: name ``drift`` type to ``trajectory`` to fit CF conventions (``drift`` still accepted as an alias) `(#599) <https://github.com/pyTMD/pyTMD/pull/599>`_
* ``test``: add tests for setting ``Dataset`` coordinate types `(#599) <https://github.com/pyTMD/pyTMD/pull/599>`_
* ``test``: ``ruff`` format some test modules `(#599) <https://github.com/pyTMD/pyTMD/pull/599>`_
* ``refactor``: handle coordinate conversion in ``_coords`` `(#599) <https://github.com/pyTMD/pyTMD/pull/599>`_
* ``docs``: add time zone recipe `(#600) <https://github.com/pyTMD/pyTMD/pull/600>`_
* ``docs``: change some icons to use ``octicons`` `(#600) <https://github.com/pyTMD/pyTMD/pull/600>`_
* ``docs``: add note about DST transitions `(#600) <https://github.com/pyTMD/pyTMD/pull/600>`_
* ``refactor``: convert time zones demo to notebook `(#601) <https://github.com/pyTMD/pyTMD/pull/601>`_
* ``docs``: add ``pandas`` demo to time zones notebook `(#601) <https://github.com/pyTMD/pyTMD/pull/601>`_
* ``docs``: use ``pandas`` for solve table `(#601) <https://github.com/pyTMD/pyTMD/pull/601>`_
* ``docs``: move units section in Getting Started `(#601) <https://github.com/pyTMD/pyTMD/pull/601>`_
* ``docs``: add some basic unit conversion examples `(#601) <https://github.com/pyTMD/pyTMD/pull/601>`_
* ``feat``: add ``pixi`` task to make clean docs `(#601) <https://github.com/pyTMD/pyTMD/pull/601>`_
* ``refactor``: create ``Dataset`` at the end of the solve fit loop `(#601) <https://github.com/pyTMD/pyTMD/pull/601>`_
* ``docs``: merge map and html table cells in solve notebook `(#602) <https://github.com/pyTMD/pyTMD/pull/602>`_
* ``docs``: add cropping demo to recipes `(#602) <https://github.com/pyTMD/pyTMD/pull/602>`_
* ``docs``: add tidal aliasing demo to recipes `(#602) <https://github.com/pyTMD/pyTMD/pull/602>`_
* ``refactor``: simplify ``ipyleaflet`` marker usage `(#602) <https://github.com/pyTMD/pyTMD/pull/602>`_
* ``docs``: add troubleshooting page to user guide `(#603) <https://github.com/pyTMD/pyTMD/pull/603>`_
* ``docs``: add chunking demo and expand crop demo `(#603) <https://github.com/pyTMD/pyTMD/pull/603>`_
* ``docs``: uniformity of header styles in ``*.rst`` files `(#603) <https://github.com/pyTMD/pyTMD/pull/603>`_
* ``docs``: clean up whitespace in ``*.rst`` files `(#603) <https://github.com/pyTMD/pyTMD/pull/603>`_
* ``refactor``: separated interpolation of admittances from inference functions `(#604) <https://github.com/pyTMD/pyTMD/pull/604>`_
* ``docs``: add mean longitude plots (S and H) `(#604) <https://github.com/pyTMD/pyTMD/pull/604>`_
* ``docs``: add lunar node cycle plot `(#604) <https://github.com/pyTMD/pyTMD/pull/604>`_
* ``test``: add check for short-term admittances `(#604) <https://github.com/pyTMD/pyTMD/pull/604>`_
* ``fix``: don't need ``p1`` in admittance function `(#604) <https://github.com/pyTMD/pyTMD/pull/604>`_
* ``fix``: use ``ValueError`` instead of bare ``Exception`` `(#604) <https://github.com/pyTMD/pyTMD/pull/604>`_
* ``docs``: reorganize API reference to new ``toctree`` `(#605) <https://github.com/pyTMD/pyTMD/pull/605>`_
* ``docs``: add buttons for API and troubleshooting `(#605) <https://github.com/pyTMD/pyTMD/pull/605>`_
* ``docs``: add JPL SPK file background page `(#605) <https://github.com/pyTMD/pyTMD/pull/605>`_
* ``feat``: add ``PYTMD_CACHE_DIR`` environ for overriding default cache `(#606) <https://github.com/pyTMD/pyTMD/pull/606>`_
* ``docs``: add a configuration section to install `(#606) <https://github.com/pyTMD/pyTMD/pull/606>`_
* ``docs``: improve ``:ref:`` usage throughout documentation `(#606) <https://github.com/pyTMD/pyTMD/pull/606>`_

.. __: https://github.com/pyTMD/pyTMD/releases/tag/3.0.9
