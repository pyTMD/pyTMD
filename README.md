# pyTMD

Python-based tidal prediction software for estimating ocean, load, solid Earth and pole tides

## About

<table>
  <tr>
    <td><b>Version:</b></td>
    <td>
        <a href="https://pypi.python.org/pypi/pyTMD/" alt="PyPI"><img src="https://img.shields.io/pypi/v/pyTMD.svg"></a>
        <a href="https://anaconda.org/conda-forge/pytmd" alt="conda-forge"><img src="https://img.shields.io/conda/vn/conda-forge/pytmd"></a>
        <a href="https://github.com/pyTMD/pyTMD/releases/latest" alt="commits-since"><img src="https://img.shields.io/github/commits-since/pyTMD/pyTMD/latest"></a>
    </td>
  </tr>
  <tr>
    <td><b>Citation:</b></td>
    <td>
        <a href="https://doi.org/10.21105/joss.08566" alt="JOSS"><img src="https://joss.theoj.org/papers/10.21105/joss.08566/status.svg"></a>
        <a href="https://doi.org/10.5281/zenodo.5555395" alt="zenodo"><img src="https://zenodo.org/badge/DOI/10.5281/zenodo.5555395.svg"></a>
    </td>
  </tr>
  <tr>
    <td><b>Tests:</b></td>
    <td>
        <a href="https://pytmd.readthedocs.io/en/latest/?badge=latest" alt="Documentation Status"><img src="https://readthedocs.org/projects/pytmd/badge/?version=latest"></a>
        <a href="https://github.com/pyTMD/pyTMD/actions/workflows/python-request.yml" alt="Build"><img src="https://github.com/pyTMD/pyTMD/actions/workflows/python-request.yml/badge.svg"></a>
        <a href="https://github.com/pyTMD/pyTMD/actions/workflows/ruff-format.yml" alt="Ruff"><img src="https://github.com/pyTMD/pyTMD/actions/workflows/ruff-format.yml/badge.svg"></a>
    </td>
  </tr>
  <tr>
    <td><b>Data:</b></td>
    <td>
        <a href="https://doi.org/10.5281/zenodo.18091740" alt="zenodo"><img src="https://img.shields.io/badge/zenodo-pyTMD_test_data-2f6fa7.svg?logo=data:image/svg%2bxml;base64,PHN2ZyBpZD0iU3ZnanNTdmcxMDIxIiB3aWR0aD0iMjg4IiBoZWlnaHQ9IjI4OCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB2ZXJzaW9uPSIxLjEiIHhtbG5zOnhsaW5rPSJodHRwOi8vd3d3LnczLm9yZy8xOTk5L3hsaW5rIiB4bWxuczpzdmdqcz0iaHR0cDovL3N2Z2pzLmNvbS9zdmdqcyI+PGRlZnMgaWQ9IlN2Z2pzRGVmczEwMjIiPjwvZGVmcz48ZyBpZD0iU3ZnanNHMTAyMyI+PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIGVuYWJsZS1iYWNrZ3JvdW5kPSJuZXcgMCAwIDIyMCA4MCIgdmlld0JveD0iMCAwIDUxLjA0NiA1MS4wNDYiIHdpZHRoPSIyODgiIGhlaWdodD0iMjg4Ij48cGF0aCBmaWxsPSIjZmZmZmZmIiBkPSJtIDI4LjMyNCwyMC4wNDQgYyAtMC4wNDMsLTAuMTA2IC0wLjA4NCwtMC4yMTQgLTAuMTMxLC0wLjMyIC0wLjcwNywtMS42MDIgLTEuNjU2LC0yLjk5NyAtMi44NDgsLTQuMTkgLTEuMTg4LC0xLjE4NyAtMi41ODIsLTIuMTI1IC00LjE4NCwtMi44MDUgLTEuNjA1LC0wLjY3OCAtMy4zMDksLTEuMDIgLTUuMTA0LC0xLjAyIC0xLjg1LDAgLTMuNTY0LDAuMzQyIC01LjEzNywxLjAyIC0xLjQ2NywwLjYyOCAtMi43NjQsMS40ODggLTMuOTEsMi41NTIgViAxNC44NCBjIDAsLTEuNTU3IC0xLjI2MiwtMi44MjIgLTIuODIsLTIuODIyIGggLTE5Ljc3NSBjIC0xLjU1NywwIC0yLjgyLDEuMjY1IC0yLjgyLDIuODIyIDAsMS41NTkgMS4yNjQsMi44MiAyLjgyLDIuODIgaCAxNS41NDEgbCAtMTguMjMsMjQuNTQ2IGMgLTAuMzYyLDAuNDg3IC0wLjU1NywxLjA3NyAtMC41NTcsMS42ODIgdiAxLjg0MSBjIDAsMS41NTggMS4yNjQsMi44MjIgMi44MjIsMi44MjIgSCA1LjAzOCBjIDEuNDg4LDAgMi43MDUsLTEuMTUzIDIuODEyLC0yLjYxNCAwLjkzMiwwLjc0MyAxLjk2NywxLjM2NCAzLjEwOSwxLjg0OCAxLjYwNSwwLjY4NCAzLjI5OSwxLjAyMSA1LjEwMiwxLjAyMSAyLjcyMywwIDUuMTUsLTAuNzI2IDcuMjg3LC0yLjE4NyAxLjcyNywtMS4xNzYgMy4wOTIsLTIuNjM5IDQuMDg0LC00LjM4OSAwLjgzMjc5OSwtMS40NzIwOTQgMS40MTgyODQsLTIuNjMzMzUyIDEuMjIxODg5LC0zLjcyOTE4MiAtMC4xNzMwMDMsLTAuOTY1MzE4IC0wLjY5NDkxNCwtMS45NDY0MTkgLTIuMzI2ODY1LC0yLjM3ODM1OCAtMC41OCwwIC0xLjM3NjAyNCwwLjE3NDU0IC0xLjgzMzAyNCwwLjQ5MjU0IC0wLjQ2MywwLjMxNiAtMC43OTMsMC43NDQgLTAuOTgyLDEuMjc1IGwgLTAuNDUzLDAuOTMgYyAtMC42MzEsMS4zNjUgLTEuNTY2LDIuNDQzIC0yLjgwOSwzLjI0NCAtMS4yMzgsMC44MDMgLTIuNjMzLDEuMjAxIC00LjE4OCwxLjIwMSAtMS4wMjMsMCAtMi4wMDQsLTAuMTkxIC0yLjk1NSwtMC41NzkgLTAuOTQxLC0wLjM5IC0xLjc1OCwtMC45MzUgLTIuNDM5LC0xLjY0IEMgOS45ODYsNDAuMzQzIDkuNDQxLDM5LjUyNiA5LjAyNywzOC42MDMgOC42MTcsMzcuNjc5IDguNDEsMzYuNzEgOC40MSwzNS42ODcgdiAtMi40NzYgaCAxNy43MTUgYyAwLDAgMS41MTc3NzQsLTAuMTU0NjYgMi4xODMzNzUsLTAuNzcwNjcyIDAuOTU4NDk2LC0wLjg4NzA4NSAwLjg2NDYyMiwtMi4xNTAzOCAwLjg2NDYyMiwtMi4xNTAzOCAwLDAgLTAuMDQzNTQsLTUuMDY2ODM0IC0wLjMzODM3NiwtNy41NzgxNTQgQyAyOC43MjkwNDgsMjEuODEyNTYzIDI4LjMyNCwyMC4wNDQgMjguMzI0LDIwLjA0NCBaIE0gLTExLjc2Nyw0Mi45MSAyLjk5MSwyMy4wMzYgQyAyLjkxMywyMy42MjMgMi44NywyNC4yMiAyLjg3LDI0LjgyNyB2IDEwLjg2IGMgMCwxLjc5OSAwLjM1LDMuNDk4IDEuMDU5LDUuMTA0IDAuMzI4LDAuNzUyIDAuNzE5LDEuNDU4IDEuMTU2LDIuMTE5IC0wLjAxNiwwIC0wLjAzMSwtMTBlLTQgLTAuMDQ3LC0xMGUtNCBIIC0xMS43NjcgWiBNIDIzLjcxLDI3LjY2NyBIIDguNDA5IHYgLTIuODQxIGMgMCwtMS4wMTUgMC4xODksLTEuOTkgMC41OCwtMi45MTIgMC4zOTEsLTAuOTIyIDAuOTM2LC0xLjc0IDEuNjQ1LC0yLjQ0NCAwLjY5NywtMC43MDMgMS41MTYsLTEuMjQ5IDIuNDM4LC0xLjY0MSAwLjkyMiwtMC4zODggMS45MiwtMC41ODEgMi45OSwtMC41ODEgMS4wMiwwIDIuMDAyLDAuMTkzIDIuOTQ5LDAuNTgxIDAuOTQ5LDAuMzkzIDEuNzY0LDAuOTM4IDIuNDQxLDEuNjQxIDAuNjgyLDAuNzA0IDEuMjI1LDEuNTIxIDEuNjQxLDIuNDQ0IDAuNDE0LDAuOTIyIDAuNjE3LDEuODk2IDAuNjE3LDIuOTEyIHoiIHRyYW5zZm9ybT0idHJhbnNsYXRlKDIwLjM1IC00LjczNSkiIGNsYXNzPSJjb2xvcmZmZiBzdmdTaGFwZSI+PC9wYXRoPjwvc3ZnPjwvZz48L3N2Zz4="></a>
        <a href="https://doi.org/10.6084/m9.figshare.30260326" alt="figshare"><img src="https://img.shields.io/badge/figshare-pyTMD_test_data-a60845?logo=figshare"></a>
    </td>
  </tr>
  <tr>
    <td><b>License:</b></td>
    <td>
        <a href="https://github.com/pyTMD/pyTMD/blob/main/LICENSE" alt="License"><img src="https://img.shields.io/github/license/pyTMD/pyTMD"></a>
    </td>
  </tr>
</table>

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

>  T. C. Sutterley, S. L. Howard, L. Padman, and M. R. Siegfried,
> "pyTMD: Python-based tidal prediction software". *Journal of Open Source Software*,
> 10(116), 8566, (2025). [doi: 10.21105/joss.08566](https://doi.org/10.21105/joss.08566)
> 
> T. C. Sutterley, T. Markus, T. A. Neumann, M. R. van den Broeke, J. M. van Wessem, and S. R. M. Ligtenberg,
> "Antarctic ice shelf thickness change from multimission lidar mapping", *The Cryosphere*,
> 13, 1801-1817, (2019). [doi: 10.5194/tc-13-1801-2019](https://doi.org/10.5194/tc-13-1801-2019)
>
> L. Padman, M. R. Siegfried, and H. A. Fricker,
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
