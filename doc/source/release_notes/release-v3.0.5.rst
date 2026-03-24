.. _release-v3.0.5:

##################
`Release v3.0.5`__
##################

* ``fix``: output data as a ``DataArray`` from ``extrapolate`` for `#537 <https://github.com/pyTMD/pyTMD/issues/537>`_ `(#538) <https://github.com/pyTMD/pyTMD/pull/538>`_
* ``feat``: allow caching of the KD-tree during extrapolation `(#540) <https://github.com/pyTMD/pyTMD/pull/540>`_
* ``refactor``: switch method for Legendre polynomials `(#541) <https://github.com/pyTMD/pyTMD/pull/541>`_
* ``test``: compare ``Plm`` and ``dPlm`` against prior values `(#541) <https://github.com/pyTMD/pyTMD/pull/541>`_
* ``fix``: handle the ``dPlm`` singularity at the poles `(#542) <https://github.com/pyTMD/pyTMD/pull/542>`_
* ``docs``: add geocentric latitude section `(#543) <https://github.com/pyTMD/pyTMD/pull/543>`_
* ``refactor``: clean up ``ephemerides`` method of calculating solid earth tides `(#543) <https://github.com/pyTMD/pyTMD/pull/543>`_
* ``ci``: bump versions for ``nodejs`` deprecation `(#543) <https://github.com/pyTMD/pyTMD/pull/543>`_
* ``ci``: smaller coverage reports `(#543) <https://github.com/pyTMD/pyTMD/pull/543>`_
* ``feat``: add tide-generating forces from Tamura (1982) `(#544) <https://github.com/pyTMD/pyTMD/pull/544>`_
* ``docs``: standardize how units are represented `(#544) <https://github.com/pyTMD/pyTMD/pull/544>`_
* ``refactor``: add more constituents to length of day `(#544) <https://github.com/pyTMD/pyTMD/pull/544>`_
* ``refactor``: make maximum degree (``lmax``) an argument `(#544) <https://github.com/pyTMD/pyTMD/pull/544>`_
* ``test``: add comparison of Legendre polynomials vs HW95 `(#544) <https://github.com/pyTMD/pyTMD/pull/544>`_
* ``docs``: standardize case of docstrings `(#545) <https://github.com/pyTMD/pyTMD/pull/545>`_
* ``refactor``: clean up ephemerides corrections for SE tides `(#546) <https://github.com/pyTMD/pyTMD/pull/546>`_
* ``refactor``: use Cartesian approach for TGF `(#547) <https://github.com/pyTMD/pyTMD/pull/547>`_
* ``feat``: new options for lunisolar ECEF XYZ `(#547) <https://github.com/pyTMD/pyTMD/pull/547>`_
* ``test``: add tests over all approximate lunisolar ECEF methods `(#547) <https://github.com/pyTMD/pyTMD/pull/547>`_

.. __: https://github.com/pyTMD/pyTMD/releases/tag/3.0.5
