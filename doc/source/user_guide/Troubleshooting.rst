.. _troubleshooting:

===============
Troubleshooting
===============

Output Is Invalid
=================

- **Check 1:** Are the coordinates over land or outside the model domain?

Most ocean tide models only have data over the ocean, which may not extend to all coastal areas.
Any point interpolated over land, inland water, or beyond the model's geographic boundary would return as ``NaN``.
For points at or near the coast: can try using one of the :ref:`extrapolation options <extrapolate-demo>`.

.. warning::

    Extrapolated values have degraded accuracy, and may lead to misleading results.
    Both bathymetry and the extrapolation distance need to be carefully considered when using these options.

Additionally, some models are purely regional and cover only part of the globe.
For points outside the domain of a regional model: probably need to use a different model.
Post on the ``pyTMD`` `discussions board <https://github.com/orgs/pyTMD/discussions>`_ if you want help choosing a model.

- **Check 2:** Are the coordinates in a different projection or :ref:`coordinate reference system <spatial-coordinates>`?

Some models are defined in a projected coordinate system rather than geographic coordinates (longitude, latitude).
Other models are defined in geographic coordinates but use a different longitudinal convention (e.g. ``[-180, 180]`` vs. ``[0, 360]``).
In either case: transform the coordinates to the model coordinate reference system.

.. code-block:: python

    m = pyTMD.io.model().from_database("GOT4.10_nc")
    ds = m.open_dataset()
    x, y = ds.tmd.transform_as(lon, lat, crs=4326)

Output Looks Inaccurate
=======================

- **Check 1:** Is the time standard and zone correct?

The most common cause of a systematic phase shift is passing times in the wrong :ref:`time zone <time-zones>` or :ref:`time standard <time-standards>`.
``pyTMD`` uses times in UTC for its astronomical calculations, and so either condition would cause the predicted tides to be out of phase with the observed tides.

- **Check 2:** Are the coordinates in a different projection or coordinate reference system?

Similar to the case that returned ``NaN``, the returned values may be finite but mapped to an incorrect location.
Verify your coordinates are in the model's :ref:`coordinate reference system <spatial-coordinates>`.

- **Check 3:** Is the extrapolation distance too large?

If extrapolating over an extended distance (e.g. ``np.inf``), the returned values may be interpolated from a location far from the intended point [see :ref:`interpolation` and :ref:`Extrapolation <extrapolate-demo>`].

- **Check 4:** Are you comparing against the correct units and reference frame?

``pyTMD`` uses a :ref:`default set of units <units>` for displacements, currents and transports.
For all predictions, the output ``xarray.DataArray`` should have attributes describing the data variable, including their units.
Check that the units of the output variables match those expected.

Additionally, for ocean and load tides, unless there is a ``z0`` term, the default :term:`tide datum <Tidal Datum>` will be in reference to the mean level and not a local datum.
For solid Earth tides, the default :ref:`tide system <permanent-tide>` is ``tide-free`` and not ``mean-tide`` as ``pyTMD`` tries to follow IERS conventions where possible.
Check that the reference frame and datum of the output variables matches those expected.

Model Not Found
===============

If the model files are not found, try the following checks:

- **Check 1:** Is the model in the :ref:`database <model-database>`?
- **Check 2:** Are the model files :ref:`downloaded locally <data-access>` or accessible from the :ref:`cloud <cloud-access>`?
- **Check 3:** Is the :ref:`directory <directories>` set correctly (if using a custom model directory)?
- **Check 4:** Does the directory path exist and contain the expected files? [see :ref:`Elevation Directory Table <directories>` and :ref:`Current Directory Table <tab-currents>`]

Predictions Are Slow
====================

If the predictions are taking a long time (or there are memory issues), try the following fixes:

- **Fix 1:** :ref:`Crop <cropping-demo>` the model to the region of interest
- **Fix 2:** Read the model in :ref:`chunks <chunking-demo>`
- **Fix 3:** Use :ref:`nearest-neighbour interpolation <interpolation>`
- **Fix 4:** Reduce the :ref:`constituent <tab-constituents>` list (will likely reduce accuracy)

Getting Help
============

If this page does not resolve your problem:

1. **Search** existing `Discussions <https://github.com/orgs/pyTMD/discussions>`_ and `Issues <https://github.com/pyTMD/pyTMD/issues>`_
2. **Open** a new `Discussion <https://github.com/orgs/pyTMD/discussions/new/choose>`_ for usage questions or `Issue <https://github.com/pyTMD/pyTMD/issues/new>`_  for potential problems [see :ref:`Guidelines <contributing>`]
