.. _release-v2.2.7:

##################
`Release v2.2.7`__
##################

* ``docs``: published rendered versions of notebooks for `#394 <https://github.com/pyTMD/pyTMD/issues/394>`_ `(#432) <https://github.com/pyTMD/pyTMD/pull/432>`_
* ``refactor``: move NOAA webservices API query functions to module `(#433) <https://github.com/pyTMD/pyTMD/pull/433>`_
* ``feat``: add simplified body tides from `Cartwright and Tayler (1971) <http://dx.doi.org/10.1111/j.1365-246X.1971.tb01803.x>`_ `(#436) <https://github.com/pyTMD/pyTMD/pull/436>`_
* ``ci``: bump python version to ``3.13`` `(#436) <https://github.com/pyTMD/pyTMD/pull/436>`_
* ``chore``: bump scipy version to ``1.16.0`` `(#436) <https://github.com/pyTMD/pyTMD/pull/436>`_
* ``docs``: improve comments on simplified body tides
* ``docs``: add link to contribution guidelines for `#434 <https://github.com/pyTMD/pyTMD/issues/434>`_ `(#437) <https://github.com/pyTMD/pyTMD/pull/437>`_
* ``docs``: update tidal spectra to add degree-3 tides `(#437) <https://github.com/pyTMD/pyTMD/pull/437>`_
* ``docs``: set images to latest ``ubuntu`` and ``python`` `(#438) <https://github.com/pyTMD/pyTMD/pull/438>`_
* ``fix``: replace invalid NOAA water level values `(#439) <https://github.com/pyTMD/pyTMD/pull/439>`_
* ``feat``: add anelastic correction for long-period body tides `(#439) <https://github.com/pyTMD/pyTMD/pull/439>`_
* ``test``: add check for long-period complex love numbers `(#439) <https://github.com/pyTMD/pyTMD/pull/439>`_
* ``feat``: Basic pixi setup `(#435) <https://github.com/pyTMD/pyTMD/pull/435>`_
* ``refactor``: unpin dependencies in ``pyproject.toml`` `(#440) <https://github.com/pyTMD/pyTMD/pull/440>`_
* ``docs``: add pixi to install instructions `(#440) <https://github.com/pyTMD/pyTMD/pull/440>`_
* ``refactor``: drop ``setuptools-scm`` dependency `(#441) <https://github.com/pyTMD/pyTMD/pull/441>`_
* ``refactor``: move scripts to be entry points `(#441) <https://github.com/pyTMD/pyTMD/pull/441>`_
* ``docs``: fix urls for moved scripts `(#442) <https://github.com/pyTMD/pyTMD/pull/442>`_
* ``fix``: argparse of entry points in docs `(#442) <https://github.com/pyTMD/pyTMD/pull/442>`_
* ``docs``: pixi as subsection on ``README.rst`` `(#443) <https://github.com/pyTMD/pyTMD/pull/443>`_
* ``feat``: update functions for diurnal complex love numbers `(#444) <https://github.com/pyTMD/pyTMD/pull/444>`_
* ``feat``: include mantle anelastic effects when inferring long-period tides `(#444) <https://github.com/pyTMD/pyTMD/pull/444>`_
* ``feat``: add option for mantle anelasticity for LPET predictions `(#444) <https://github.com/pyTMD/pyTMD/pull/444>`_
* ``refactor``: switch time decimal in pole tides to nominal years of 365.25 days `(#445) <https://github.com/pyTMD/pyTMD/pull/445>`_
* ``refactor``: convert angles with numpy ``radians`` and ``degrees`` functions `(#445) <https://github.com/pyTMD/pyTMD/pull/445>`_
* ``feat``: add ``arcs2rad`` and ``rad2arcs`` functions and use in conversions `(#445) <https://github.com/pyTMD/pyTMD/pull/445>`_
* ``refactor``: rename to ``asec2rad`` `(#446) <https://github.com/pyTMD/pyTMD/pull/446>`_
* ``fix``: return numpy arrays if cannot infer minor constituents `(#447) <https://github.com/pyTMD/pyTMD/pull/447>`_
* ``fix``: make long-period inference optional `(#447) <https://github.com/pyTMD/pyTMD/pull/447>`_
* ``refactor``: convert microarcseconds to radians with ``masec2rad`` function in ``math.py`` `(#448) <https://github.com/pyTMD/pyTMD/pull/448>`_

.. __: https://github.com/pyTMD/pyTMD/releases/tag/2.2.7
