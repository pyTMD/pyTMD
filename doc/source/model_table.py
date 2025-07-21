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
for model,parameters in models[model_type].items():
    # extract the model directory
    if isinstance(parameters['model_file'], str):
        model_directory = pathlib.Path(parameters['model_file']).parent
    elif isinstance(parameters['model_file'], list):
        model_directory = pathlib.Path(parameters['model_file'][0]).parent
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
for model,parameters in models[model_type].items():
    # extract the model directory
    model_directories = []
    for t in parameters['model_file'].keys():
        if isinstance(parameters['model_file'][t], str):
            d = pathlib.Path(parameters['model_file'][t]).parent
        elif isinstance(parameters['model_file'][t], list):
            d = pathlib.Path(parameters['model_file'][t][0]).parent
        model_directories.append(f'``<path_to_tide_models>/{d}``')
    # join the directories
    model_directory = ','.join(model_directories)
    # extract the reference
    reference = parameters.get('reference', None)
    # write the model and directories to the csv file
    fid.write(f'`{model} <{reference}>`_,{model_directory}\n')
# close the file
fid.close()
