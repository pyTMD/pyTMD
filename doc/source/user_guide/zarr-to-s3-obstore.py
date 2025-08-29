import pyTMD
import zarr
import obstore

# set tide model and s3 bucket
s3_bucket = 'pytmd-scratch'
tide_model = 'FES2022'
# setup tide model
m = pyTMD.io.model(compressed=False)
# read tide model and convert to xarray DataTree
dtree = m.to_datatree(tide_model, gap_fill=True)

# setup zarr store using obstore
presigned_s3_url = f's3://{s3_bucket}/{m.name}.zarr'
s3_store = obstore.store.S3Store.from_url(presigned_s3_url, region="us-west-2")
store = zarr.storage.ObjectStore(s3_store, read_only=False)
# save to zarr store
dtree.to_zarr(store, mode='w', zarr_format=3, consolidated=True)
