"""Create a table of available tidal models
"""
import pathlib
from pyTMD.io import load_database
# documentation directory
directory = pathlib.Path(__file__).parent
# create model table
models_table = directory.joinpath('_assets', 'models.csv')
fid = models_table.open(mode='w', encoding='utf8')
models = load_database()
# write to csv
print('Model,Directory', file=fid)
for model,parameters in models['elevation'].items():
    if isinstance(parameters['model_file'], str):
        directory = pathlib.Path(parameters['model_file']).parent
    elif isinstance(parameters['model_file'], list):
        directory = pathlib.Path(parameters['model_file'][0]).parent
    reference = parameters.get('reference', None)
    print(f'`{model} <{reference}>`_,``<path_to_tide_models>/{directory}``', file=fid)
fid.close()
