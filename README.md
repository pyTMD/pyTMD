# pyTMD

[![License](https://img.shields.io/github/license/pyTMD/pyTMD)](https://github.com/pyTMD/pyTMD/blob/main/LICENSE)
[![Documentation Status](https://readthedocs.org/projects/pytmd/badge/?version=latest)](https://pytmd.readthedocs.io/en/latest/?badge=latest)
[![PyPI](https://img.shields.io/pypi/v/pyTMD.svg)](https://pypi.python.org/pypi/pyTMD/)
[![conda-forge](https://img.shields.io/conda/vn/conda-forge/pytmd)](https://anaconda.org/conda-forge/pytmd)
[![commits-since](https://img.shields.io/github/commits-since/pyTMD/pyTMD/latest)](https://github.com/pyTMD/pyTMD/releases/latest)
[![zenodo](https://zenodo.org/badge/DOI/10.5281/zenodo.5555395.svg)](https://doi.org/10.5281/zenodo.5555395)

Python-based tidal prediction software for estimating ocean, load, solid Earth and pole tides

For more information: see the documentation at [pytmd.readthedocs.io](https://pytmd.readthedocs.io/)

## Installation

From PyPI:

```bash
python3 -m pip install pyTMD
```

To include all optional dependencies:

```bash
python3 -m pip install pyTMD[all]
```

Using `conda` or `mamba` from conda-forge:

```bash
conda install -c conda-forge pytmd
```

```bash
mamba install -c conda-forge pytmd
```

Development version from GitHub:

```bash
python3 -m pip install git+https://github.com/pyTMD/pyTMD.git
```

### Running with Pixi

Alternatively, you can use [Pixi](https://pixi.sh/) for a streamlined workspace environment:

1. Install Pixi following the [installation instructions](https://pixi.sh/latest/#installation)
2. Clone the project repository:

```bash
git clone https://github.com/pyTMD/pyTMD.git
```

3. Move into the `pyTMD` directory

```bash
cd pyTMD
```

4. Install dependencies and start JupyterLab:

```bash
pixi run start
```

This will automatically create the environment, install all dependencies, and launch JupyterLab in the [notebooks](./doc/source/notebooks/) directory.

## Dependencies

- [h5netcdf: Pythonic interface to netCDF4 via h5py](https://h5netcdf.org/)
- [lxml: processing XML and HTML in Python](https://pypi.python.org/pypi/lxml)
- [numpy: Scientific Computing Tools For Python](https://www.numpy.org)
- [platformdirs: Python module for determining platform-specific directories](https://pypi.org/project/platformdirs/)
- [pyproj: Python interface to PROJ library](https://pypi.org/project/pyproj/)
- [scipy: Scientific Tools for Python](https://www.scipy.org/)
- [timescale: Python tools for time and astronomical calculations](https://pypi.org/project/timescale/)
- [xarray: N-D labeled arrays and datasets in Python](https://docs.xarray.dev/en/stable/) 

## References

> T. C. Sutterley, T. Markus, T. A. Neumann, M. R. van den Broeke, J. M. van Wessem, and S. R. M. Ligtenberg,
> "Antarctic ice shelf thickness change from multimission lidar mapping", *The Cryosphere*,
> 13, 1801-1817, (2019). [doi: 10.5194/tc-13-1801-2019](https://doi.org/10.5194/tc-13-1801-2019)
>
> L. Padman, M. R. Siegfried, H. A. Fricker,
> "Ocean Tide Influences on the Antarctic and Greenland Ice Sheets", *Reviews of Geophysics*,
> 56, 142-184, (2018). [doi: 10.1002/2016RG000546](https://doi.org/10.1002/2016RG000546)

## Download

The program homepage is:  
<https://github.com/pyTMD/pyTMD>

A zip archive of the latest version is available directly at:  
<https://github.com/pyTMD/pyTMD/archive/main.zip>

## Alternative Software

perth5 from NASA Goddard Space Flight Center:  
<https://codeberg.org/rray/perth5>

Matlab Tide Model Driver from Earth & Space Research:  
<https://github.com/EarthAndSpaceResearch/TMD_Matlab_Toolbox_v2.5>

Fortran OSU Tidal Prediction Software:  
<https://www.tpxo.net/otps>

## Disclaimer

This package includes software developed at NASA Goddard Space Flight Center (GSFC) and the University of Washington Applied Physics Laboratory (UW-APL).
It is not sponsored or maintained by the Universities Space Research Association (USRA), AVISO or NASA.
The software is provided here for your convenience but *with no guarantees whatsoever*.
It should not be used for coastal navigation or any application that may risk life or property.

## Contributing

This project contains work and contributions from the [scientific community](./CONTRIBUTORS.md).
If you would like to contribute to the project, please have a look at the [contribution guidelines](./doc/source/getting_started/Contributing.rst), [open issues](https://github.com/pyTMD/pyTMD/issues) and [discussions board](https://github.com/pyTMD/pyTMD/discussions).

## Credits

The Tidal Model Driver ([TMD](https://github.com/EarthAndSpaceResearch/TMD_Matlab_Toolbox_v2.5)) Matlab Toolbox was developed by Laurie Padman, Lana Erofeeva and Susan Howard.
An updated version of the TMD Matlab Toolbox ([TMD3](https://github.com/chadagreene/Tide-Model-Driver)) was developed by Chad Greene.
The OSU Tidal Inversion Software (OTIS) and OSU Tidal Prediction Software ([OTPS](https://www.tpxo.net/otps)) were developed by Lana Erofeeva and Gary Egbert ([copyright OSU](https://www.tpxo.net/tpxo-products-and-registration), licensed for non-commercial use).
The NASA Goddard Space Flight Center (GSFC) PREdict Tidal Heights (PERTH3) software was developed by Richard Ray and Remko Scharroo.
An updated and more versatile version of the NASA GSFC tidal prediction software ([PERTH5](https://codeberg.org/rray/perth5)) was developed by Richard Ray.

## License

The content of this project is licensed under the [Creative Commons Attribution 4.0 Attribution license](https://creativecommons.org/licenses/by/4.0/) and the source code is licensed under the [MIT license](LICENSE).
