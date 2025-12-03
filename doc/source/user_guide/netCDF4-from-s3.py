import pyTMD

# set tide model and s3 bucket
s3_bucket = 'pytmd-scratch'
tide_model = 'GOT5.5'
# s3 bucket url
s3_url = f"https://{s3_bucket}.s3.us-west-2.amazonaws.com"
# setup tide model
m = pyTMD.io.model(s3_url).from_database(tide_model)
# read tide model dataset
ds = m.open_dataset(group='z', chunks='auto')
