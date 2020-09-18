compute_LPET_elevation.py
=========================

 - Calculates long-period equilibrium tides for an input file
 - Uses the summation of fifteen tidal spectral lines from [Cartwright and Edden, (1973)](https://doi.org/10.1111/j.1365-246X.1973.tb03420.x)

#### Calling Sequence
```bash
python compute_LPET_elevation.py input_file output_file
```
[Source code](https://github.com/tsutterley/pyTMD/blob/master/compute_LPET_elevation.py)

#### Inputs
 1. `input_file`: name of input file
 2. `output_file`: name of output file

#### Command Line Options
- `--format=X`: input and output data format
    * `'csv'` (default)
    * `'netCDF4'`
    * `'HDF5'`
- `--variables=X`: variable names of data in csv, HDF5 or netCDF4 file
    * for csv files: the order of the columns within the file
    * for HDF5 and netCDF4 files: time, y, x and data variable names
- `--epoch=X`: Reference epoch of input time
    * `'days since 1858-11-17T00:00:00'` (default Modified Julian Days)
- `--projection=X`: spatial projection as EPSG code or PROJ4 string
    * `4326`: latitude and longitude coordinates on WGS84 reference ellipsoid
- `-V`, `--verbose`: Verbose output of processing run
 - `-M X`, `--mode=X`: Permission mode of output file
