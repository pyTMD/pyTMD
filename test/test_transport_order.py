#!/usr/bin/env python
u"""
test_transport_order.py
Tests the order of operations when converting depth-averaged transports
to currents using a small synthetic ATLAS-netcdf model

Dividing by bathymetry on the model grid before interpolating
differs from interpolating transports and bathymetry and then dividing
(as done in OTPS and TMD2.5) when the interpolation stencil contains
very different depths
"""
import json
import warnings
import numpy as np
import xarray as xr
import pytest
import pyTMD

# synthetic u-cell: one deep and one very shallow corner
LON = np.array([10.0, 11.0])
LAT = np.array([-30.0, -29.0])
# depths (ny, nx) and complex transports (m^2/s)
DEPTH = np.array([[33.2, 33.2], [33.2, 2.9]])
TRANSPORT = np.array([[1.0 + 0.5j, 1.0 + 0.5j], [1.0 + 0.5j, 0.1 - 0.2j]])
XI, YI = 10.5, -29.5


def _write_model(path):
    """Write minimal ATLAS-netcdf grid and M2 transport files"""
    dims = ("ny", "nx")
    grid = xr.Dataset()
    for g in ("z", "u", "v"):
        grid[f"lon_{g}"] = ("nx", LON)
        grid[f"lat_{g}"] = ("ny", LAT)
        grid[f"h{g}"] = (dims, DEPTH.copy(), {"units": "meter"})
    grid.to_netcdf(path / "grid.nc")
    for g in ("u", "v"):
        ds = xr.Dataset()
        ds["con"] = np.array(list("m2  "), dtype="|S1")
        ds[f"lon_{g}"] = ("nx", LON)
        ds[f"lat_{g}"] = ("ny", LAT)
        ds[f"{g}Re"] = (dims, TRANSPORT.real.copy(), {"units": "m^2/s"})
        ds[f"{g}Im"] = (dims, TRANSPORT.imag.copy(), {"units": "m^2/s"})
        ds.to_netcdf(path / f"{g}_m2.nc")
    # z file for completeness
    ds = xr.Dataset()
    ds["con"] = np.array(list("m2  "), dtype="|S1")
    ds["lon_z"] = ("nx", LON)
    ds["lat_z"] = ("ny", LAT)
    ds["hRe"] = (dims, np.ones((2, 2)), {"units": "mm"})
    ds["hIm"] = (dims, np.zeros((2, 2)), {"units": "mm"})
    ds.to_netcdf(path / "h_m2.nc")


@pytest.fixture(scope="module")
def model(tmp_path_factory):
    path = tmp_path_factory.mktemp("atlas")
    _write_model(path)
    d = dict(
        format="ATLAS-netcdf", name="synthetic", compressed=False,
        z=dict(model_file=[str(path / "h_m2.nc")],
            grid_file=str(path / "grid.nc"), units="mm"),
        u=dict(model_file=[str(path / "u_m2.nc")],
            grid_file=str(path / "grid.nc"), units="m^2/s"),
        v=dict(model_file=[str(path / "v_m2.nc")],
            grid_file=str(path / "grid.nc"), units="m^2/s"),
    )
    return pyTMD.io.model().from_dict(d)


def _interp(m, **kwargs):
    ds = m.open_dataset(group="u", **kwargs)
    return ds.tmd.interp(np.array([XI]), np.array([YI]))


def test_divide_before_is_default(model):
    local = _interp(model)
    # on-grid division: bilinear mean of U/h in cm/s
    expected = (TRANSPORT / DEPTH).mean() * 100.0
    assert np.isclose(local["m2"].values.item(), expected)


def test_divide_after(model):
    local = _interp(model, transport=True)
    # transports in the dataset are unchanged
    assert local["m2"].tmd.group == "transport"
    out = local.tmd.to_currents().tmd.to_default_units()
    expected = TRANSPORT.mean() / DEPTH.mean() * 100.0
    assert out["m2"].tmd.group == "current"
    assert np.isclose(out["m2"].values.item(), expected)
    # the two orders must differ for this stencil
    before = _interp(model)["m2"].values.item()
    assert abs(out["m2"].values.item() - before) > 0.1 * abs(before)


def test_min_depth(model):
    local = _interp(model, transport=True)
    out = local.tmd.to_currents(min_depth=100.0).tmd.to_default_units()
    expected = TRANSPORT.mean() / 100.0 * 100.0
    assert np.isclose(out["m2"].values.item(), expected)


def test_deep_cells_agree(model):
    # at a deep node both orders are identical
    ds = model.open_dataset(group="u")
    ds2 = model.open_dataset(group="u", transport=True)
    x, y = np.array([LON[0]]), np.array([LAT[0]])
    a = ds.tmd.interp(x, y)["m2"].values.item()
    b = ds2.tmd.interp(x, y).tmd.to_currents().tmd.to_default_units()
    assert np.isclose(a, b["m2"].values.item())


def test_requires_bathymetry(model):
    ds = model.open_dataset(group="u").drop_vars("bathymetry")
    with pytest.raises(ValueError):
        ds.tmd.to_currents()


def test_unsupported_format():
    m = pyTMD.io.model().from_dict(dict(
        format="GOT-netcdf", name="x", z=dict(model_file=["a"], units="cm"),
        u=dict(model_file=["a"], units="cm/s")), validate=False)
    with pytest.raises(ValueError):
        m.open_dataset(group="u", transport=True)


def test_compute_tide_currents(tmp_path):
    # write model files and relative-path definition file
    _write_model(tmp_path)
    definition = tmp_path / "synthetic.json"
    params = dict(format="ATLAS-netcdf", name="synthetic", compressed=False)
    for g, f in (("z", "h_m2.nc"), ("u", "u_m2.nc"), ("v", "v_m2.nc")):
        params[g] = dict(model_file=[f], grid_file="grid.nc", units="m^2/s")
    definition.write_text(json.dumps(params))
    kwargs = dict(
        directory=tmp_path, definition_file=definition, type="drift",
        standard="datetime", infer_minor=False,
    )
    x, y = np.array([XI]), np.array([YI])
    t = np.array(["2020-01-01T00:00"], dtype="datetime64[ns]")
    before = pyTMD.compute.tide_currents(x, y, t, **kwargs)
    after = pyTMD.compute.tide_currents(x, y, t, transport=True, **kwargs)
    assert set(after.keys()) == {"u", "v"}
    assert not np.isclose(before["u"].values, after["u"].values)
    # a large minimum depth should reduce the amplitude
    floor = pyTMD.compute.tide_currents(
        x, y, t, transport=True, min_depth=1000.0, **kwargs
    )
    assert abs(floor["u"].values.item()) < abs(after["u"].values.item())


def _params(path, transport_groups=("U", "V")):
    """Model parameters with transports in the given groups"""
    d = dict(format="ATLAS-netcdf", name="synthetic", compressed=False)
    d["z"] = dict(model_file=[str(path / "h_m2.nc")],
        grid_file=str(path / "grid.nc"), units="mm")
    for g in transport_groups:
        d[g] = dict(model_file=[str(path / f"{g.lower()}_m2.nc")],
            grid_file=str(path / "grid.nc"), units="m^2/s")
    return d


def test_legacy_groups_are_moved(tmp_path):
    # lowercase transports (legacy definitions) are moved to U and V
    m = pyTMD.io.model().from_dict(_params(tmp_path, ("u", "v")), validate=False)
    assert all(g in m.__dict__ for g in ("U", "V"))
    assert not any(g in m.__dict__ for g in ("u", "v"))
    assert m.to_dict()["U"] is m.U
    # legacy attribute access still works, but warns
    with pytest.warns(DeprecationWarning, match="'U' group"):
        assert m.u is m.U
    with pytest.warns(DeprecationWarning, match="'V' group"):
        assert m["v"] is m["V"]
    # uppercase groups do not warn
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert m["U"].units == "m^2/s"
        assert m.V.units == "m^2/s"


def test_currents_groups_are_kept():
    # true currents (not transports) stay in lowercase groups
    d = dict(format="FES-netcdf", name="x",
        z=dict(model_file="a", units="cm"), u=dict(model_file="a", units="cm/s"),
        v=dict(model_file="a", units="cm/s"))
    m = pyTMD.io.model().from_dict(d, validate=False)
    assert all(g in m.__dict__ for g in ("u", "v"))
    assert not any(g in m.__dict__ for g in ("U", "V"))
    with pytest.raises(ValueError):
        m.open_dataset(group="U")


def test_database_transports():
    # transports in the database are in the uppercase groups
    m = pyTMD.io.model(verify=False).from_database("CATS2008")
    assert m["U"].units == "m^2/s" and m["V"].units == "m^2/s"
    assert not any(g in m.__dict__ for g in ("u", "v"))
    # FES currents are in the lowercase groups
    m = pyTMD.io.model(verify=False).from_database("FES2014")
    assert "u" in m.__dict__ and "U" not in m.__dict__
    assert "CATS2008" in pyTMD.io.model.ocean_current()
    assert "FES2014" in pyTMD.io.model.ocean_current()


@pytest.mark.parametrize("groups", [("U", "V"), ("u", "v")])
def test_open_uppercase_groups(tmp_path, groups):
    _write_model(tmp_path)
    m = pyTMD.io.model().from_dict(_params(tmp_path, groups), validate=False)
    ds_u = m.open_dataset(group="u")
    ds_U = m.open_dataset(group="U")
    ds_t = m.open_dataset(group="u", transport=True)
    assert ds_u["m2"].tmd.group == "current"
    assert ds_U["m2"].tmd.group == "transport"
    xr.testing.assert_identical(ds_U, ds_t)
    # datatree keys follow the requested groups
    dtree = m.open_datatree(group=["u", "v"])
    assert set(dtree.children) == {"u", "v"}
    assert dtree["u"]["m2"].tmd.group == "current"
    dtree = m.open_datatree(group=["U", "V"])
    assert dtree["U"]["m2"].tmd.group == "transport"


def test_no_internal_deprecation_warnings(tmp_path):
    # reading models should not use the legacy aliases internally
    _write_model(tmp_path)
    for groups in (("U", "V"), ("u", "v")):
        with warnings.catch_warnings():
            warnings.simplefilter("error", DeprecationWarning)
            m = pyTMD.io.model().from_dict(_params(tmp_path, groups))
            m.open_datatree(group=["z", "u", "v"])
            m.open_datatree(group=["U", "V"])
            m.open_dataset(group="u", transport=True)
            m.to_dict(serialize=True)
            _ = m.multifile


@pytest.mark.parametrize("groups", [("U", "V"), ("u", "v")])
def test_parse_and_reduce_constituents(tmp_path, groups):
    # parsing and reducing constituents of transports should not warn
    for c in ("m2", "s2"):
        (tmp_path / f"u_{c}_tpxo9_atlas_30_v2.nc").touch()
    d = _params(tmp_path, groups)
    for g in groups:
        d[g]["model_file"] = [
            str(tmp_path / f"u_{c}_tpxo9_atlas_30_v2.nc") for c in ("m2", "s2")
        ]
    m = pyTMD.io.model().from_dict(d, validate=False)
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        for g in ("u", "U"):
            m.parse_constituents(group=g)
            assert m.constituents == ["m2", "s2"]
        m.reduce_constituents(["s2"], group="u")
        assert m.constituents == ["s2"]
        assert len(m["U"].model_file) == 1
