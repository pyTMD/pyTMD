import pyTMD
import zarr
import s3fs

# set tide model and s3 bucket
s3_bucket = 'pytmd-scratch'
tide_model = 'FES2022'
# setup tide model
m = pyTMD.io.model().from_database(tide_model)

# setup zarr store using s3fs
fs = s3fs.S3FileSystem(anon=False, asynchronous=True)
store = zarr.storage.FsspecStore(fs, path=f'{s3_bucket}/{m.name}.zarr')
