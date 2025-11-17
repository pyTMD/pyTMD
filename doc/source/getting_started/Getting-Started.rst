===============
Getting Started
===============

.. tip::

    See the `background material <../background/Tides.html>`_ and `glossary <../background/Glossary.html>`_ for more information on the theory and methods used in ``pyTMD``.

Tide Model Formats
##################

Ocean and load tide constituent files are available from different modeling groups in different formats.
``pyTMD`` can access the harmonic constituents for the OTIS, GOT and FES families of ocean and load tide models.
OTIS and ATLAS formatted data use binary files to store the constituent data for either heights (``z``) or zonal and meridional transports (``u``, ``v``).
They can be either a single file containing all the constituents (compact) or multiple files each containing a single constituent.
ATLAS netCDF formatted data use netCDF4 files for each constituent and variable type (``z``, ``u``, ``v``).
GOT formatted data use ascii or netCDF4 files for each height constituent (``z``).
FES formatted data use either ascii (1999, 2004) or netCDF4 (2012, 2014) files for each constituent and variable type (``z``, ``u``, ``v``).
``pyTMD`` uses ``pint`` to handle the units of the model constituent data and convert them into standard sets of units.

    - ``z``: tidal elevations in meters (:math:`m`)
    - ``u``: zonal tidal currents in centimeters per second (:math:`cm/s`)
    - ``U``: zonal tidal transports in square meters per second (:math:`m^2/s`)
    - ``v``: meridional tidal currents in centimeters per second (:math:`cm/s`)
    - ``V``: meridional tidal transports in square meters per second (:math:`m^2/s`)

Data Access
###########

Some tide models can be programmatically downloaded using the fetching routines in ``pyTMD.datasets``.
OTIS-formatted Arctic Ocean models can be downloaded from the NSF ArcticData server using the :func:`pyTMD.datasets.fetch_arcticdata` function.
GOT models can be downloaded from the NASA GSFC server using the :func:`pyTMD.datasets.fetch_gsfc_got` function.
Users registered with AVISO [see :ref:`aviso-registration`] can download FES models from their FTP server using the :func:`pyTMD.datasets.fetch_aviso_fes` function.

Other tide models may require manual downloading due to licensing agreements or limitations on programmatic access.
TPXO models (OTIS and ATLAS formats) can be requested from the data producers after `registration <https://www.tpxo.net/tpxo-products-and-registration>`_.
OTIS-formatted Antarctic models are available from the U.S. Antarctic Program Data Center (USAP-DC), which uses a reCAPTCHA security system to prevent automated access.
See the model links in :ref:`directories` for the references to specific tide models.

Model Database
##############

``pyTMD`` comes parameterized with models for the prediction of tidal elevations and currents.
All presently available models are stored within a `JSON database <https://github.com/pyTMD/pyTMD/blob/main/pyTMD/data/database.json>`_:

.. code-block:: python

   >>> import pyTMD
   >>> pyTMD.models.get('CATS2008')
   {'format': 'OTIS', 'name': 'CATS2008', 'projection': {'datum': 'WGS84', 'lat_0': -90, 'lat_ts': -71, 'lon_0': -70, 'proj': 'stere', 'type': 'crs', 'units': 'km', 'x_0': 0, 'y_0': 0}, 'reference': 'https://doi.org/10.15784/601235', 'u': {'grid_file': 'CATS2008/grid_CATS2008', 'model_file': 'CATS2008/uv.CATS2008.out', 'units': 'm^2/s', 'variable': 'zonal_tidal_current'}, 'v': {'grid_file': 'CATS2008/grid_CATS2008', 'model_file': 'CATS2008/uv.CATS2008.out', 'units': 'm^2/s', 'variable': 'meridional_tidal_current'}, 'z': {'grid_file': 'CATS2008/grid_CATS2008', 'model_file': 'CATS2008/hf.CATS2008.out', 'units': 'm', 'variable': 'tide_ocean'}}

``pyTMD`` currently supports several solutions from the following tide models:

- Arctic Ocean (AO) and Greenland coast (Gr) tidal simulations :cite:p:`Padman:2004hv`
- Circum-Antarctic Tidal Simulations (CATS) :cite:p:`Padman:2008ec`
- Empirical Ocean Tide (EOT) models :cite:p:`HartDavis:2021dx`
- Finite Element Solution (FES) tide models :cite:p:`Lyard:2021fk`
- Goddard Ocean Tide (GOT) models :cite:p:`Ray:1999vm`
- Hamburg direct data Assimilation Methods for Tides (HAMTIDE) models :cite:p:`Taguchi:2014ht`
- Technical University of Denmark (DTU) tide models :cite:p:`Andersen:2023ei`
- TOPEX/POSEIDON (TPXO) global tide models :cite:p:`Egbert:2002ge`

.. _directories:

Directories
###########

``pyTMD`` uses a tree structure for storing and accessing the tidal constituent data.
This structure was chosen based on the different formats of each tide model.
The base of the tree structure (in the table below as ``<path_to_tide_models>``) can be the default internal directory (``pyTMD/data``) or a user-specified (external) directory.
Several models can be programmatically downloaded from their providers to their parameterized directories using the fetching routines in ``pyTMD.datasets``.

Presently, the following models and their directories are parameterized within ``pyTMD``:

.. csv-table::
   :file: ../_assets/elevation-models.csv
   :header-rows: 1

.. tip::
    See :ref:`tab-currents` for the table of directories for models with tidal currents. 

For other tide models, the model parameters can be set with a `model definition file <./Getting-Started.html#definition-files>`_.
If you wish to add a new model to the ``pyTMD`` database, please see the `contribution guidelines <./Contributing.html>`_.

.. note::
    Any model parameterized with a definition file or added to the database will have to fit a presently supported file standard.

Definition Files
################

For models not currently within the ``pyTMD`` `database <./Getting-Started.html#model-database>`_, the model parameters can be set in :py:class:`pyTMD.io.model` with a definition file in JSON format.
The JSON definition files follow a similar structure as the main ``pyTMD`` database, but for individual entries.
The JSON format directly maps the parameter names with their values stored in the appropriate data type (strings, lists, numbers, booleans, etc).
While still human readable, the JSON format is both interoperable and more easily machine readable.

Each definition file should have ``name``, ``format`` and ``type`` parameters.
Each model type may also require specific sets of parameters for the individual model reader.
For models with multiple constituent files, the files can be found using a ``glob`` string to search a directory.

- ``OTIS``, ``ATLAS-compact`` and ``TMD3``

    * ``format``: ``OTIS``, ``ATLAS-compact`` or ``TMD3``
    * ``name``: tide model name
    * ``projection``: `model spatial projection <./Getting-Started.html#spatial-coordinates>`_.
    * ``z``:

        - ``grid_file``: path to model grid file
        - ``model_file``: path to model constituent file(s) or a ``glob`` string
        - ``units``: units of the model constituent data
    * ``u``:

        - ``grid_file``: path to model grid file
        - ``model_file``: path to model constituent file(s) or a ``glob`` string
        - ``units``: units of the model constituent data
    * ``v``:

        - ``grid_file``: path to model grid file
        - ``model_file``: path to model constituent file(s) or a ``glob`` string
        - ``units``: units of the model constituent data

- ``ATLAS-netcdf``

    * ``format``: ``ATLAS-netcdf``
    * ``name``: tide model name
    * ``compressed``: model files are ``gzip`` compressed
    * ``z``:

        - ``grid_file``: path to model grid file
        - ``model_file``: path to model constituent files or a ``glob`` string
        - ``units``: units of the model constituent data
    * ``u``:

        - ``grid_file``: path to model grid file
        - ``model_file``: path to model constituent files or a ``glob`` string
        - ``units``: units of the model constituent data
    * ``v``:

        - ``grid_file``: path to model grid file
        - ``model_file``: path to model constituent files or a ``glob`` string
        - ``units``: units of the model constituent data

- ``GOT-ascii`` and ``GOT-netcdf``

    * ``format``: ``GOT-ascii`` or ``GOT-netcdf``
    * ``name``: tide model name
    * ``compressed``: model files are ``gzip`` compressed
    * ``z``:

        - ``grid_file``: path to model grid file
        - ``model_file``: path to model constituent files or a ``glob`` string
        - ``units``: units of the model constituent data

- ``FES-ascii`` and ``FES-netcdf``

    * ``compressed``: model files are ``gzip`` compressed
    * ``format``: ``FES-ascii`` or ``FES-netcdf``
    * ``name``: tide model name
    * ``version``: tide model version
    * ``z``:

        - ``grid_file``: path to model grid file
        - ``model_file``: path to model constituent files or a ``glob`` string
        - ``units``: units of the model constituent data
    * ``u``:

        - ``grid_file``: path to model grid file
        - ``model_file``: path to model constituent files or a ``glob`` string
        - ``units``: units of the model constituent data
    * ``v``:

        - ``grid_file``: path to model grid file
        - ``model_file``: path to model constituent files or a ``glob`` string
        - ``units``: units of the model constituent data

Programs
########

``pyTMD.compute`` calculates tide predictions for use with ``numpy`` arrays or ``pandas`` dataframes.
These are a series of functions that take ``x``, ``y``, and ``time`` coordinates and
compute the corresponding tidal elevation or currents.

.. code-block:: python

    >>> import pyTMD
    >>> tide_h = pyTMD.compute.tide_elevations(x, y, delta_time, DIRECTORY=path_to_tide_models, MODEL='CATS2008', EPSG=3031, EPOCH=(2000,1,1,0,0,0), TYPE='drift', TIME='GPS', METHOD='linear', FILL_VALUE=np.nan)
    >>> tide_uv = pyTMD.compute.tide_currents(x, y, delta_time, DIRECTORY=path_to_tide_models, MODEL='CATS2008', EPSG=3031, EPOCH=(2000,1,1,0,0,0), TYPE='drift', TIME='GPS', METHOD='linear', FILL_VALUE=np.nan)

Time
####

The default time in ``pyTMD`` is days (UTC) since a given epoch.
For ocean, load and equilibrium tide programs, the epoch is 1992-01-01T00:00:00.
For pole tide programs, the epoch is 1858-11-17T00:00:00 (Modified Julian Days).
``pyTMD`` uses the ``timescale`` library to convert different time formats to the necessary time format of a given program.
``timescale`` can also parse date strings describing the units and epoch of relative times, or the calendar date of measurement for geotiff formats.

Spatial Coordinates
###################

The default coordinate system in ``pyTMD`` is WGS84 geodetic coordinates in latitude and longitude.
``pyTMD`` uses ``pyproj`` to convert from different coordinate systems and datums.
Some regional tide models are projected in a different coordinate system.
These models have their coordinate reference system (CRS) information stored as PROJ descriptors in the `JSON model database <https://github.com/pyTMD/pyTMD/blob/main/pyTMD/data/database.json>`_:
For other projected models, a formatted coordinate reference system (CRS) descriptor (e.g. ``PROJ``, ``WKT``, or ``EPSG`` code) can be used.

Interpolation
#############

For converting from model coordinates, ``pyTMD`` uses spatial interpolation routines from ``xarray``.
For coastal or near-grounded points, the model can be extrapolated with :func:`pyTMD.interpolate.extrapolate` using a nearest-neighbor routine.
The default maximum extrapolation distance is 10 kilometers.
This default distance may not be a large enough extrapolation for some applications and models.

.. warning::
    The extrapolation cutoff can be set to any distance in kilometers, but should be used with caution in cases such as narrow fjords or ice sheet grounding zones :cite:p:`Padman:2018cv`.
