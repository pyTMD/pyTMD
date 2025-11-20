import zarr
import pyTMD
import obstore
import timescale
import pandas as pd
import xarray as xr

# set tide model and s3 bucket
s3_bucket = 'pytmd-scratch'
tide_model = 'FES2022'
# setup tide model
m = pyTMD.io.model(verify=False).from_database(tide_model)

# setup s3 store
presigned_s3_url = f's3://{s3_bucket}/{m.name}.zarr'
s3_store = obstore.store.S3Store.from_url(presigned_s3_url,
    region="us-west-2", skip_signature=True)
# use read_only store for accessing data
store = zarr.storage.ObjectStore(s3_store, read_only=True)

# read zarr store for tide model
ds = xr.open_zarr(store, group='z', zarr_format=3)

# read data from parquet
df = pd.read_parquet('pytmd-test.parquet')
ts = timescale.from_deltatime(df.time, epoch=(2018,1,1), standard='GPS')

# convert points to crs of model
x, y = ds.tmd.transform(df.x, df.y, crs=3031)
# create xarray DataArrays for coordinates
x = xr.DataArray(x, dims='time')
y = xr.DataArray(y, dims='time')
# interpolate to points
local = ds.tmd.interp(x, y, method='linear')

# predict tides and infer minor constituents
df[m.z.variable] = local.tmd.predict(ts.tide,
    deltat=ts.tt_ut1, corrections=m.corrections)
df[m.z.variable] += local.tmd.infer(ts.tide,
    deltat=ts.tt_ut1, corrections=m.corrections)

# save model outputs to parquet
df.to_parquet(f'{m.name}.parquet')
