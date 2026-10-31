import pyTMD
import xarray as xr
from pyTMD.utilities import get_cache_path

# read tide model
model = "GOT4.8"
m = pyTMD.io.model().from_database(model)
ds = m.open_dataset(group="z")

# temporary output dataset to create output zarr
tmp = xr.Dataset({c: xr.zeros_like(ds[c]) for c in ds.tmd.constituents})
# convert to dataset with amplitude and phase
z = tmp.tmd.to_dataarray()
output = xr.Dataset({"amp": z.tmd.amplitude, "ph": z.tmd.phase})
# use distributed writes to the zarr store
zarr_filename = get_cache_path(f"{m.name}-load-GF.zarr")
output.to_zarr(zarr_filename, mode="w", compute=False, consolidated=False)

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
ds["rho_w"] = filled["rho_w"]

# calculate grid cell areas
ds["area"] = ds.tmd.cell_area()

# calculate X, Y and Z from geodetic latitude and longitude
X, Y, Z = pyTMD.spatial.to_cartesian(lon, lat)
ds["X"] = (("y", "x"), X)
ds["Y"] = (("y", "x"), Y)
ds["Z"] = (("y", "x"), Z)

# create dataset for output coordinates
XYZ1 = ds[["X", "Y", "Z"]]

# flatten tidal dataset and remove masked points
XYZ2 = ds.stack(xy=("x", "y")).dropna("xy")
XYZ2 = XYZ2.reset_index("xy")

# get flattened density and area
rho_w = XYZ2["rho_w"]
area = XYZ2["area"]
# drop variables from flattened dataset
XYZ2 = XYZ2.drop_vars(("rho_w", "area"))

# for each x-coordinate in the output map
for x in XYZ1.x:
    # calculate deformation
    tmp = pyTMD.solve.potential._greens_function(
        XYZ1.sel(x=[x]),
        XYZ2,
        area=area,
        rho_w=rho_w,
        lln="wang-prem",
        reference="CF",
        use_cache=True,
    )
    # major constituents as a dataarray
    z = tmp.tmd.to_dataarray()
    # write slice to zarr
    output = xr.Dataset({"amp": z.tmd.amplitude, "ph": z.tmd.phase})
    output.to_zarr(zarr_filename, region="auto", consolidated=False)
