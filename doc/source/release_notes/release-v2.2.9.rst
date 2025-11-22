##################
`Release v2.2.9`__
##################

* ``refactor``: generalize the calculation of body tides for degrees 3+ `(#467) <https://github.com/pyTMD/pyTMD/pull/467>`_
* ``docs``: added IERS Conventions references for Love number calculations `(#467) <https://github.com/pyTMD/pyTMD/pull/467>`_
* ``docs``: add a draft of a JOSS paper `(#415) <https://github.com/pyTMD/pyTMD/pull/415>`_
* ``ci``: add JOSS paper build `(#415) <https://github.com/pyTMD/pyTMD/pull/415>`_
* ``docs``: add non-antarctic pyTMD use cases for `#416 <https://github.com/pyTMD/pyTMD/issues/416>`_  `(#415) <https://github.com/pyTMD/pyTMD/pull/415>`_
* ``docs``: add cite to ESA cci+ velocities `(#415) <https://github.com/pyTMD/pyTMD/pull/415>`_
* ``docs``: add doi for Kantha and Clayson `(#415) <https://github.com/pyTMD/pyTMD/pull/415>`_
* ``docs``: mention ``PERTH`` is Fortran software `(#415) <https://github.com/pyTMD/pyTMD/pull/415>`_
* ``fix``: typo `(#466) <https://github.com/pyTMD/pyTMD/pull/466>`_ `(#415) <https://github.com/pyTMD/pyTMD/pull/415>`_
* ``docs``: add blurbs about the harmonic method `(#415) <https://github.com/pyTMD/pyTMD/pull/415>`_
* ``docs``: add tide potential catalog table `(#415) <https://github.com/pyTMD/pyTMD/pull/415>`_
* ``ci``: add environmental variable for paper path `(#415) <https://github.com/pyTMD/pyTMD/pull/415>`_
* ``docs``: add spherical harmonic plots `(#415) <https://github.com/pyTMD/pyTMD/pull/415>`_
* ``docs``: add table of max tide potential `(#415) <https://github.com/pyTMD/pyTMD/pull/415>`_
* ``chore``: add more ``ruff`` lint parameters `(#415) <https://github.com/pyTMD/pyTMD/pull/415>`_
* ``refactor``: functional access to downloading scripts for `#470 <https://github.com/pyTMD/pyTMD/issues/470>`_ `(#415) <https://github.com/pyTMD/pyTMD/pull/415>`_
* ``docs``: expand on the directory structure `(#415) <https://github.com/pyTMD/pyTMD/pull/415>`_
* ``docs``: update programs for downloading from ArcticData, GOT or AVISO for `#469 <https://github.com/pyTMD/pyTMD/issues/469>`_ `(#415) <https://github.com/pyTMD/pyTMD/pull/415>`_
* ``docs``: add admonitions noting models will need to be downloaded `(#415) <https://github.com/pyTMD/pyTMD/pull/415>`_
* ``docs``: admonition in ``xarray`` recipe `(#415) <https://github.com/pyTMD/pyTMD/pull/415>`_
* ``test``: add developer download routines for `#471 <https://github.com/pyTMD/pyTMD/issues/471>`_ `(#415) <https://github.com/pyTMD/pyTMD/pull/415>`_
* ``fix``: make ``pixi`` tasks associated with features for `#472 <https://github.com/pyTMD/pyTMD/issues/472>`_ `(#415) <https://github.com/pyTMD/pyTMD/pull/415>`_
* ``docs``: update data access for `#470 <https://github.com/pyTMD/pyTMD/issues/470>`_ `(#415) <https://github.com/pyTMD/pyTMD/pull/415>`_
* ``docs``: add notes about AVISO+ registration `(#415) <https://github.com/pyTMD/pyTMD/pull/415>`_
* ``fix``: move ``xarray`` to ``all`` dependencies `(#415) <https://github.com/pyTMD/pyTMD/pull/415>`_
* ``docs``: fix ``pytest`` parallel for `#473 <https://github.com/pyTMD/pyTMD/issues/473>`_ `(#415) <https://github.com/pyTMD/pyTMD/pull/415>`_
* ``ci``: refactor to use ``pixi`` `(#415) <https://github.com/pyTMD/pyTMD/pull/415>`_
* ``ci``: fetch data from open repository `(#415) <https://github.com/pyTMD/pyTMD/pull/415>`_
* ``ci``: no longer reliant on AWS s3 `(#415) <https://github.com/pyTMD/pyTMD/pull/415>`_
* ``refactor``: remove ``environment.yml`` and add ``pixi`` task to export `(#415) <https://github.com/pyTMD/pyTMD/pull/415>`_
* ``chore``: clean up ``pyproject.toml`` `(#415) <https://github.com/pyTMD/pyTMD/pull/415>`_
* ``docs``: change build from ``micromamba`` to ``pixi`` `(#415) <https://github.com/pyTMD/pyTMD/pull/415>`_
* ``docs``: add note for fetching test data for `#471 <https://github.com/pyTMD/pyTMD/issues/471>`_ `(#415) <https://github.com/pyTMD/pyTMD/pull/415>`_
* ``ci``: drop ``macos-latest`` build |cry| `(#415) <https://github.com/pyTMD/pyTMD/pull/415>`_
* ``docs``: add acknowledgment to our JOSS reviewers `(#474) <https://github.com/pyTMD/pyTMD/pull/474>`_
* ``test``: fix ``pixi`` task for testing `(#475) <https://github.com/pyTMD/pyTMD/pull/475>`_
* ``feat``: can solve with inferred minor constituents (post-fit) `(#476) <https://github.com/pyTMD/pyTMD/pull/476>`_
* ``docs``: add ``:func:`` and ``:py:class:`` pointers `(#477) <https://github.com/pyTMD/pyTMD/pull/477>`_
* ``refactor``: remove some deprecated functions `(#477) <https://github.com/pyTMD/pyTMD/pull/477>`_
* ``feat``: added option to use memory mapping for reading large files `(#478) <https://github.com/pyTMD/pyTMD/pull/478>`_
* ``feat``: create ``fetch`` class to group fetching functions `(#479) <https://github.com/pyTMD/pyTMD/pull/479>`_
* ``fix``: assign order to ``reshape`` if not using ``mmap`` `(#479) <https://github.com/pyTMD/pyTMD/pull/479>`_
* ``refactor``: change default directory for tide models to cache `(#481) <https://github.com/pyTMD/pyTMD/pull/481>`_
* ``feat``: add ``platformdirs`` to dependencies `(#481) <https://github.com/pyTMD/pyTMD/pull/481>`_
* ``fix``: simplify ``pixi`` build with ``matplotlib-base`` `(#482) <https://github.com/pyTMD/pyTMD/pull/482>`_
* ``docs``: elaborate on Julian centuries `(#482) <https://github.com/pyTMD/pyTMD/pull/482>`_
* ``fix``: skip 4 bytes if no boundary conditions `(#482) <https://github.com/pyTMD/pyTMD/pull/482>`_
* ``feat``: added ``crs`` property for model coordinate reference system `(#483) <https://github.com/pyTMD/pyTMD/pull/483>`_
* ``fix``: include filtering mask out from possible parsed names `(#483) <https://github.com/pyTMD/pyTMD/pull/483>`_
* ``feat``: added ``crs`` property for model coordinate reference system `(#483) <https://github.com/pyTMD/pyTMD/pull/483>`_
* ``feat``: added ``is_global`` property for models covering a global domain `(#483) <https://github.com/pyTMD/pyTMD/pull/483>`_
* ``feat``: added ``pad`` function to pad global datasets along boundaries `(#483) <https://github.com/pyTMD/pyTMD/pull/483>`_
* ``feat``: added ``inpaint`` function to fill missing data in datasets `(#483) <https://github.com/pyTMD/pyTMD/pull/483>`_
* ``feat``: add projection info for all models to ``database.json`` `(#483) <https://github.com/pyTMD/pyTMD/pull/483>`_
* ``feat``: add dataarray accessor for amplitude and phase `(#483) <https://github.com/pyTMD/pyTMD/pull/483>`_

.. __: https://github.com/pyTMD/pyTMD/releases/tag/2.2.9

.. |cry|    unicode:: U+1F622 .. CRYING FACE
