=======================
Transition Guide for v3
=======================

Overview
========

The transition from ``pyTMD`` version 2 to version 3 includes several changes to the
API and functionality of the library. This document outlines the key differences
between the two versions and provides guidance on how to adapt existing code to
work with ``pyTMD`` v3.

Key Changes
===========

1. **xarray Integration**: utilizes ``xarray`` for reading, writing and operating on datasets. This improves our capability to handle multi-dimensional arrays.
2. **Unit Support**: incorporates ``pint`` for unit management, allowing for more robust handling of physical quantities and unit conversion.
3. **Database Restructuring**: The tidal model database has been restructured to flatten the hierarchy and (hopefully) improve accessibility.
4. **Function Renaming and Relocation**: Several functions have been renamed or moved to different modules.

Examples
========

**Example 1: Reading a Tidal Model and Predicting a Time Series**

.. code-block:: python
    :caption: Version 2

    import pyTMD
    # get model parameters for model
    m = pyTMD.io.model(directory).elevation(model)
    # read model and interpolate to location
    amp, ph, c = m.extract_constants(lon, lat, extrapolate=True)
    # calculate complex phase in radians for Euler's
    cph = -1j*ph*np.pi/180.0
    # calculate constituent oscillation
    hc = amp*np.exp(cph)
    # calculate tide values for time series
    tide = pyTMD.predict.time_series(time, hc, c,
        deltat=deltat, corrections=m.corrections)
    # infer minor corrections and add to prediction
    tide += pyTMD.predict.infer_minor(time, hc, c,
        deltat=deltat, corrections=m.corrections)

.. code-block:: python
    :caption: Version 3

    import pyTMD
    # get model parameters for model
    m = pyTMD.io.model(directory).from_database(model)
    # read model as xarray Dataset
    ds = m.open_dataset(group='z')
    # interpolate to location
    local = m.tmd.interp(lon, lat, extrapolate=True)
    # calculate tide values for time series
    tide = local.tmd.predict(time, deltat=deltat,
        corrections=m.corrections)
    # infer minor corrections and add to prediction
    tide += local.tmd.infer(time, deltat=deltat,
        corrections=m.corrections)

