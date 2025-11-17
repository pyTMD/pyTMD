"""Create a table of available tidal models
"""
import pathlib
from pyTMD.io import load_database

# documentation directory
directory = pathlib.Path(__file__).parent
# load the database of tidal models
models = load_database()

# create model table
model_type = 'elevation'
models_table = directory.joinpath('_assets', f'{model_type}-models.csv')
fid = models_table.open(mode='w', encoding='utf8')
# write to csv
fid.write('Model,Directory\n')
for model,parameters in models.items():
    # skip current models
    if 'z' not in parameters:
        continue
    # extract the model directory
    if isinstance(parameters['z']['model_file'], str):
        model_directory = pathlib.Path(parameters['z']['model_file']).parent
    elif isinstance(parameters['z']['model_file'], list):
        model_directory = pathlib.Path(parameters['z']['model_file'][0]).parent
    # extract the reference
    reference = parameters.get('reference', None)
    # write the model and directory to the csv file
    fid.write(f'`{model} <{reference}>`_,``<path_to_tide_models>/{model_directory}``\n')
# close the file
fid.close()

# create model table
model_type = 'current'
models_table = directory.joinpath('_assets', f'{model_type}-models.csv')
fid = models_table.open(mode='w', encoding='utf8')
# write to csv
fid.write('Model,U-Directory,V-Directory\n')
for model,parameters in models.items():
    # skip elevation models
    if 'u' not in parameters:
        continue
    # extract the model directory
    model_directories = []
    for t in ('u', 'v'):
        if isinstance(parameters[t]['model_file'], str):
            d = pathlib.Path(parameters[t]['model_file']).parent
        elif isinstance(parameters[t]['model_file'], list):
            d = pathlib.Path(parameters[t]['model_file'][0]).parent
        model_directories.append(f'``<path_to_tide_models>/{d}``')
    # only keep unique directories
    model_directories = list(set(model_directories))
    # join the directories
    model_directory = ','.join(model_directories)
    # extract the reference
    reference = parameters.get('reference', None)
    # write the model and directories to the csv file
    fid.write(f'`{model} <{reference}>`_,{model_directory}\n')
# close the file
fid.close()
