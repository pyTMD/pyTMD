.. _release-v3.0.10:

===================
`Release v3.0.10`__
===================

* ``feat``: add ``transport`` and ``min_depth`` options to ``compute.tide_currents`` to interpolate OTIS/ATLAS transports and bathymetry before converting to currents (as in ``OTPS``, ``TMD2.5`` and ``pyTMD`` v2) for `#632 <https://github.com/pyTMD/pyTMD/issues/632>`_
* ``feat``: add ``transport`` option to ``io.model.open_dataset``
* ``feat``: add ``to_currents`` method to the ``Dataset`` accessor to convert transports to currents using the bathymetry
* ``feat``: define OTIS, ATLAS and TMD3 transports in the ``U`` and ``V`` groups of the model database, and currents in ``u`` and ``v`` (lowercase transport groups in definition files are still accepted)
* ``feat``: ``io.model.open_dataset`` and ``open_datatree`` accept the ``'U'`` and ``'V'`` groups to read transports
* ``docs``: use ``U`` and ``V`` groups in model table and definition file documentation
* ``docs``: add section on currents from OTIS and ATLAS transports to the transition guide
* ``docs``: add currents check to troubleshooting page
* ``test``: add synthetic ATLAS model tests for the order of interpolation and division

.. __: https://github.com/pyTMD/pyTMD/releases/tag/3.0.10
