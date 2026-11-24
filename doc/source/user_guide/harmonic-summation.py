import pyTMD
import xarray as xr
from pyTMD.utilities import get_cache_path

# read tide model
model = "GOT4.8"
m = pyTMD.io.model().from_database(model)
ds = m.open_dataset(group="z", chunks="auto")

# read WOA23 decadal average density
woa23_filename = get_cache_path("woa23_decav_d00_04.nc")
woa23 = xr.open_dataset(woa23_filename, decode_times=False).isel(time=0)
mapping_coords = dict(lon="x", lat="y")
woa23 = woa23.rename(mapping_coords).drop_vars("time")

# transform model coordinates to lat/lon coordinates
lon, lat = ds.tmd.to_geographic(crs=4326)
# interpolate density to model grid
interp = woa23.tmd.interp(lon, lat)
# gap-fill using DCW inpainting
filled = interp.tmd.inpaint(N=10)
rho_w = filled["rho_w"].chunk("auto")

# maximum degree and order
lmax = 360
# convert ocean tides to spherical harmonics
Ylms = pyTMD.solve.ocean_harmonics(
    ds,
    lmax=lmax,
    lln="wang-prem",
    reference="CF",
    rho_w=rho_w,
)
# convert spherical harmonics to estimated crustal deformation
tmp = pyTMD.solve.crustal_loading(
    ds,
    Ylms,
    lmax=lmax,
    lln="wang-prem",
    reference="CF",
    variable="R",
)

# major constituents as a dataarray
z = tmp.tmd.to_dataarray()
# write to zarr
output = xr.Dataset({"amp": z.tmd.amplitude, "ph": z.tmd.phase})
zarr_filename = get_cache_path(f"{m.name}-load-SHA-L{lmax:d}.zarr")
output.to_zarr(zarr_filename, mode="w")
