=======================
Transition Guide for v3
=======================

Overview
========

Version 3 of ``pyTMD`` introduces significant updates to the codebase [see :ref:`release-v3.0.0`].
API and functionality of the library. This document outlines the key differences
between the two versions and provides guidance on how to adapt existing code to
work with ``pyTMD`` v3.

Key Changes
===========

1. **xarray Integration**: utilizes ``xarray`` for reading, writing and operating on datasets. This improves our capability to handle multi-dimensional arrays.
2. **Database Restructuring**: The tidal model database has been restructured to flatten the hierarchy and (hopefully) improve accessibility [see :ref:`restructure-json`].
3. **Unit Support**: incorporates ``pint`` for unit management, allowing for more robust handling of physical quantities and unit conversion.
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
    local = ds.tmd.interp(lon, lat, extrapolate=True)
    # calculate tide values for time series
    tide = local.tmd.predict(time, deltat=deltat,
        corrections=m.corrections)
    # infer minor corrections and add to prediction
    tide += local.tmd.infer(time, deltat=deltat,
        corrections=m.corrections)

**Example 2: Plotting a Tidal Model**

.. code-block:: python
    :caption: Version 2

    import pyTMD
    import matplotlib.pyplot as plt
    # get model parameters for model
    m = pyTMD.io.model(directory).elevation(model)
    # read model constituents
    c = m.read_constants()
    # get spatial bounds of model grid
    xmin, xmax = c.coords.x.min(), c.coords.x.max()
    ymin, ymax = c.coords.y.min(), c.coords.y.max()
    extent = [xmin, xmax, ymin, ymax]
    # plot model amplitude and phase
    fig, ax = plt.subplots(ncols=2)
    ax[0].imshow(c.amplitude('m2'), origin='lower',
        extent=extent, interpolation='nearest',
        cmap='viridis', vmin=0)
    ax[1].imshow(c.phase('m2'), origin='lower',
        extent=extent, interpolation='nearest',
        cmap='hsv', vmin=0, vmax=360)
    # show plot
    plt.show()

.. code-block:: python
    :caption: Version 3

    import pyTMD
    import matplotlib.pyplot as plt
    # get model parameters for model
    m = pyTMD.io.model().from_database(model)
    # read model as xarray Dataset
    ds = m.open_dataset(group='z')
    # plot model amplitude and phase
    fig, ax = plt.subplots(ncols=2)
    ds.m2.tmd.amplitude.plot(ax=ax[0], cmap='viridis', vmin=0)
    ds.m2.tmd.phase.plot(ax=ax[1], cmap='hsv', vmin=0, vmax=360)
    # show plot
    plt.show()


.. _restructure-json:

Restructuring Definition Files
==============================

In version 2, the structure of the JSON database and definition files were separated into elevation and currents models.
This was a legacy of the original design of the library (v1), which hardcoded the elevation and currents parameters.
Model files in version 2 could be a string or list for elevation models, and a dictionary of strings or lists for currents models.
In version 3, the database and example definition files have been flattened into a single structure [see :ref:`definition-files`].
In the new structure, each group (``'z'`` for elevation and ``'u'`` or ``'v'`` for currents) contains the model files and units.
Units can also be extracted from files when reading the model data if stored as metadata attributes.

Existing JSON files can be restructured to the new flattened format using a script provided in the ``test`` directory:

.. code-block:: bash


    python test/_restructure_json.py elevation_model.json

    python test/_restructure_json.py elevation_model.json currents_model.json


Units Support
=============

In version 2, units were only implicitly handled within ``pyTMD`` by setting scaling factors within the model definitions.
This could lead to some confusion when working with different tide models that used different units.
In version 3, units for different variables are explicitly handled using the ``pint`` library.
When reading model data, units are extracted from the definition files or from attributes in the data files.
The variables are then converted to default sets of units: 1) meters for elevation, 2) centimeters per second for currents and 3) meters squared per second for transports.
Datasets can also be converted to different units using the ``to_units`` method.

.. code-block:: python

    # convert elevation dataset to millimeters
    ds_z_mm = ds_z.tmd.to_units('mm')
    # convert currents datasets to meters per second
    ds_u_mps = ds_u.tmd.to_units('m/s')
    ds_v_mps = ds_v.tmd.to_units('m/s')

Removed Functions
=================

Each model format (``'ATLAS'``, ``'FES'``, ``'GOT'``, ``'OTIS'``, etc) previously had two functions for reading model constituents: 
``extract_constants`` was used to read model data and interpolate to a specific location, and ``read_constants`` was used to read the full model grid.
The ``constituents`` class stored the model constituents from ``read_constants`` and had some methods for calculating amplitude and phase.
In version 3, these functions have been removed and replaced with a unified interface using ``xarray`` datasets.
Model data can be read using the ``open_dataset`` method, which returns an ``xarray`` dataset containing all model constituents.
Interpolation to specific locations is performed using the ``interp`` method.

In version 2, tide prediction was separated into ``drift``, ``map`` and ``time_series`` functions depending on the shape of the input data.
In version 3, these have been consolidated into a single ``time_series`` method that uses ``xarray`` to handle different input shapes.

Function Renaming
=================

The ``arguments`` module has been renamed ``constituents`` to better reflect its expanded capabilities from the earliest versions.

