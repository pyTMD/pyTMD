import zarr
import pyTMD
import obstore
import timescale
import xarray as xr
import geopandas as gpd

# set tide model and s3 bucket
s3_bucket = 'pytmd-scratch'
tide_model = 'FES2022'
# setup tide model
m = pyTMD.io.model(verify=False).elevation(tide_model)

# setup s3 store
presigned_s3_url = f's3://{s3_bucket}/{m.name}.zarr'
s3_store = obstore.store.S3Store.from_url(presigned_s3_url,
    region="us-west-2", skip_signature=True)
# use read_only store for accessing data
store = zarr.storage.ObjectStore(s3_store, read_only=True)

# read zarr store for tide model
ds = xr.open_zarr(store, group='z', zarr_format=3)
constituents = ds.tmd.constituents

# read data from parquet
df = gpd.pd.read_parquet('pytmd-test.parquet')
ts = timescale.from_deltatime(df.time, epoch=(2018,1,1), standard='GPS')
# convert points to EPSG of model (default is 4326)
geometry = gpd.points_from_xy(df.x, df.y, crs=3031).to_crs(ds.tmd.crs)

# create xarray DataArrays for coordinates
x = xr.DataArray(geometry.x, dims='i')
y = xr.DataArray(geometry.y, dims='i')

# check if longitudinal convention needs to be adjusted
# only check if model is in geographic coordinates
if geometry.crs.is_geographic:
    # adjust input longitudes to be consistent with model
    if (x.min() < 0.0) & (ds.x.max() > 180.0):
        # input points convention (-180:180)
        # tide model convention (0:360)
        x[x < 0.0] += 360.0
    elif (x.max() > 180.0) & (ds.x.min() < 0.0):
        # input points convention (0:360)
        # tide model convention (-180:180)
        x[x > 180.0] -= 360.0

# interpolate to points and convert to DataArray
hc = ds.interp(x=x, y=y, method='linear', kwargs={"fill_value": None})
hc = hc.tmd.to_dataarray()

# predict tides and infer minor constituents
df[m.variable] = pyTMD.predict.drift(ts.tide, hc, constituents,
    deltat=ts.tt_ut1, corrections=m.corrections)
df[m.variable] += pyTMD.predict.infer_minor(ts.tide, hc, constituents,
    deltat=ts.tt_ut1, corrections=m.corrections)

# save model outputs to parquet
df.to_parquet(f'{m.name}.parquet')
