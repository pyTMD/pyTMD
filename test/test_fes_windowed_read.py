"""
test_fes_windowed_read.py
Tests index hyperslabs for regional tide-model windows.

Preferred path: lazy open (chunks) then ``Dataset.tmd.crop`` /
``DataTree.tmd.crop`` — not driver-specific open-time ``bounds``.
"""

from __future__ import annotations

import pathlib

import numpy as np
import pytest
import xarray as xr

import pyTMD.io.FES as FES
from pyTMD.io.dataset import isel_bounds


def _write_synthetic_fes_nc(path: pathlib.Path) -> None:
    """Write a tiny FES2014/2022-shaped z constituent file (m2)."""
    lon = np.arange(0.0, 10.0, 1.0)
    lat = np.arange(-5.0, 5.0, 1.0)
    amp = np.ones((lat.size, lon.size), dtype=np.float32)
    pha = np.zeros((lat.size, lon.size), dtype=np.float32)
    ds = xr.Dataset(
        data_vars={
            "amplitude": (("lat", "lon"), amp),
            "phase": (("lat", "lon"), pha),
        },
        coords={"lon": lon, "lat": lat},
    )
    # scipy/NETCDF3 keeps fixtures backend-agnostic (no HDF5 writer required)
    ds.to_netcdf(path, engine="scipy", format="NETCDF3_CLASSIC")


def _gridded_ds() -> xr.Dataset:
    x = np.arange(0.0, 10.0, 1.0)
    y = np.arange(-5.0, 5.0, 1.0)
    return xr.Dataset(
        {"m2": (("y", "x"), np.ones((y.size, x.size), dtype=np.complex64))},
        coords={"x": x, "y": y},
    )


def test_isel_bounds_windows_grid():
    ds = xr.Dataset(
        {"amplitude": (("lat", "lon"), np.ones((10, 10)))},
        coords={
            "lon": np.arange(0.0, 10.0, 1.0),
            "lat": np.arange(-5.0, 5.0, 1.0),
        },
    )
    out = isel_bounds(ds, "lon", "lat", bounds=[2.0, 4.0, -1.0, 1.0])
    np.testing.assert_array_equal(out["lon"].values, np.array([2.0, 3.0, 4.0]))
    np.testing.assert_array_equal(out["lat"].values, np.array([-1.0, 0.0, 1.0]))


def test_isel_bounds_empty_returns_empty_slices():
    ds = _gridded_ds()
    out = isel_bounds(ds, "x", "y", bounds=[100.0, 110.0, 50.0, 60.0])
    assert out.sizes["x"] == 0
    assert out.sizes["y"] == 0


def test_crop_windows_gridded_dataset():
    ds = _gridded_ds()
    out = ds.tmd.crop([2.0, 4.0, -1.0, 1.0], buffer=0)
    np.testing.assert_allclose(out["x"].values, [2.0, 3.0, 4.0])
    np.testing.assert_allclose(out["y"].values, [-1.0, 0.0, 1.0])
    assert out["m2"].sizes["x"] == 3
    assert out["m2"].sizes["y"] == 3


def test_crop_respects_buffer():
    ds = _gridded_ds()
    out = ds.tmd.crop([3.0, 3.0, 0.0, 0.0], buffer=1.0)
    np.testing.assert_allclose(out["x"].values, [2.0, 3.0, 4.0])
    np.testing.assert_allclose(out["y"].values, [-1.0, 0.0, 1.0])


def test_crop_keeps_dask_chunks():
    dask = pytest.importorskip("dask")
    del dask  # presence check only
    ds = _gridded_ds().chunk({"x": 2, "y": 2})
    assert ds.chunks is not None
    out = ds.tmd.crop([2.0, 4.0, -1.0, 1.0], buffer=0)
    # must not force chunk(-1).compute() — regional window stays lazy
    assert out.chunks is not None
    assert out["m2"].sizes["x"] == 3
    assert out["m2"].sizes["y"] == 3


def test_datatree_crop_windows_each_group():
    z = _gridded_ds()
    u = _gridded_ds().rename({"m2": "m2"})
    dtree = xr.DataTree.from_dict({"z": z, "u": u})
    out = dtree.tmd.crop([2.0, 4.0, -1.0, 1.0], buffer=0)
    for key in ("z", "u"):
        ds = out[key].to_dataset()
        np.testing.assert_allclose(ds["x"].values, [2.0, 3.0, 4.0])
        np.testing.assert_allclose(ds["y"].values, [-1.0, 0.0, 1.0])


def test_open_fes_netcdf_then_crop_windows(tmp_path):
    path = tmp_path / "m2_fes2022.nc"
    _write_synthetic_fes_nc(path)
    # chunked open keeps amp/phase→complex lazy so crop can hyperslab
    full = FES.open_fes_netcdf(path, group="z", chunks={})
    windowed = full.tmd.crop([2.0, 4.0, -1.0, 1.0], buffer=0)
    assert "m2" in full.data_vars
    assert "m2" in windowed.data_vars
    assert full["m2"].sizes["x"] == 10
    assert full["m2"].sizes["y"] == 10
    assert windowed["m2"].sizes["x"] == 3
    assert windowed["m2"].sizes["y"] == 3
    np.testing.assert_allclose(windowed["x"].values, [2.0, 3.0, 4.0])
    np.testing.assert_allclose(windowed["y"].values, [-1.0, 0.0, 1.0])


def _global_grid(x: np.ndarray) -> xr.Dataset:
    """1° global grid with encoded x-index so wrap source is visible."""
    y = np.array([-1.0, 0.0, 1.0])
    # value = original x coordinate (before pad) for wrap checks
    data = np.broadcast_to(x[np.newaxis, :], (y.size, x.size)).astype(
        np.float64
    )
    return xr.Dataset({"v": (("y", "x"), data.copy())}, coords={"x": x, "y": y})


def _crs(lon_wrap: int) -> dict:
    return {
        "proj": "longlat",
        "datum": "WGS84",
        "ellps": "WGS84",
        "lon_wrap": lon_wrap,
        "type": "crs",
    }


def test_crop_pacific_0_360_crosses_prime_meridian():
    """0–360 FES-like: [-10, 10] → dual hyperslab, continuous x, no half-globe."""
    ds = _global_grid(np.arange(0.0, 360.0, 1.0))
    ds.attrs["crs"] = _crs(180)
    out = ds.tmd.crop([-10.0, 10.0, -1.0, 1.0], buffer=0)
    np.testing.assert_allclose(out["x"].values[0], -10.0)
    np.testing.assert_allclose(out["x"].values[-1], 10.0)
    assert out.sizes["x"] == 21
    # must not pad half the globe then crop
    assert out.sizes["x"] < 50
    west = out.sel(x=-5.0)["v"].values
    np.testing.assert_allclose(west, 355.0)


def test_crop_pacific_0_360_western_basin_shifted():
    """Negative western-Pacific box maps into model 0–360 without wrap split."""
    ds = _global_grid(np.arange(0.0, 360.0, 1.0))
    ds.attrs["crs"] = _crs(180)
    out = ds.tmd.crop([-170.0, -150.0, -1.0, 1.0], buffer=0)
    assert out.sizes["x"] == 21
    # data from model longitudes 190–210
    np.testing.assert_allclose(out.sel(x=-160.0)["v"].values, 200.0)


def test_crop_pacific_0_360_dateline_crossing_xmin_gt_xmax():
    """xmin>xmax means eastward wrap: 170E → 170W (=190°E), short arc."""
    ds = _global_grid(np.arange(0.0, 360.0, 1.0))
    ds.attrs["crs"] = _crs(180)
    out = ds.tmd.crop([170.0, -170.0, -1.0, 1.0], buffer=0)
    assert out.sizes["x"] == 21
    np.testing.assert_allclose(out["x"].values[0], 170.0)
    np.testing.assert_allclose(out["x"].values[-1], 190.0)
    np.testing.assert_allclose(out.sel(x=180.0)["v"].values, 180.0)


def test_crop_pacific_m180_180_dateline_crossing():
    """−180–180 grid: dateline crossing via xmin>xmax."""
    ds = _global_grid(np.arange(-180.0, 180.0, 1.0))
    ds.attrs["crs"] = _crs(0)
    out = ds.tmd.crop([170.0, -170.0, -1.0, 1.0], buffer=0)
    assert out.sizes["x"] == 21
    np.testing.assert_allclose(out["x"].values[0], 170.0)
    np.testing.assert_allclose(out["x"].values[-1], 190.0)
    # wrapped western piece carries model x=-175 data at continuous 185
    np.testing.assert_allclose(out.sel(x=185.0)["v"].values, -175.0)


def test_crop_pacific_0_360_bounds_past_360():
    """Bounds past 360 on a 0–360 grid: east pad equivalent via dual isel."""
    ds = _global_grid(np.arange(0.0, 360.0, 1.0))
    ds.attrs["crs"] = _crs(180)
    out = ds.tmd.crop([350.0, 370.0, -1.0, 1.0], buffer=0)
    np.testing.assert_allclose(out["x"].values[0], 350.0)
    np.testing.assert_allclose(out["x"].values[-1], 370.0)
    assert out.sizes["x"] == 21
    np.testing.assert_allclose(out.sel(x=370.0)["v"].values, 10.0)


def test_crop_pacific_wrap_keeps_dask_chunks():
    pytest.importorskip("dask")
    ds = _global_grid(np.arange(0.0, 360.0, 1.0)).chunk({"x": 30, "y": 1})
    ds.attrs["crs"] = _crs(180)
    out = ds.tmd.crop([-10.0, 10.0, -1.0, 1.0], buffer=0)
    assert out.chunks is not None
    assert out.sizes["x"] == 21


def test_crop_large_buffer_does_not_force_full_globe():
    """Oversized buffer with xmin<=xmax: linear grid intersect, not full lon."""
    ds = _global_grid(np.arange(0.0, 360.0, 1.0))
    ds.attrs["crs"] = _crs(180)
    # [0, 10] + buffer 200 → [-200, 210]; historical where kept ~[0, 210]
    out = ds.tmd.crop([0.0, 10.0, -1.0, 1.0], buffer=200.0)
    assert out.sizes["x"] == 211
    np.testing.assert_allclose(out["x"].values[0], 0.0)
    np.testing.assert_allclose(out["x"].values[-1], 210.0)


def test_is_global_when_meridian_duplicated():
    """FES-style 0…360 inclusive endpoints are still global."""
    # 1° cells with both 0 and 360 present (span == 360, not 360-dx)
    x = np.arange(0.0, 361.0, 1.0)
    ds = _global_grid(x)
    assert ds.tmd.is_global


def test_crop_dateline_on_duplicated_meridian_grid():
    """Dateline crop must work when is_global uses 0…360 inclusive span."""
    x = np.arange(0.0, 361.0, 1.0)
    ds = _global_grid(x)
    ds.attrs["crs"] = _crs(180)
    out = ds.tmd.crop([170.0, -170.0, -1.0, 1.0], buffer=0)
    assert out.sizes["x"] == 21
    np.testing.assert_allclose(out["x"].values[0], 170.0)
    np.testing.assert_allclose(out["x"].values[-1], 190.0)
