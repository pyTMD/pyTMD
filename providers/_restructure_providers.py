"""
_restructure_providers.py (08/2024)
Restructure provider JSON files into new format
"""
import json
import inspect
import pathlib
import argparse

# current file path
filename = inspect.getframeinfo(inspect.currentframe()).filename
filepath = pathlib.Path(filename).absolute().parent

# PURPOSE: create argument parser
def arguments():
    parser = argparse.ArgumentParser(
        description="""Restructure provider JSON files into new format"
            """,
        fromfile_prefix_chars="@"
    )
    # command line parameters
    parser.add_argument('--pretty', '-p',
        action='store_true',
        help='Pretty print the json file')
    return parser

def main():
    # Read the system arguments listed after the program
    parser = arguments()
    args,_ = parser.parse_known_args()

    # find providers
    providers = [f for f in filepath.iterdir() if (f.suffix == '.json')]
    # variable long names for tidal currents
    long_name = dict(u='zonal_tidal_current', v='meridional_tidal_current')

    # for each provider
    for provider in providers:
        # input dictionary
        d = dict(elevation={}, current={})
        with provider.open('r', encoding='utf-8') as fid:
            p = json.load(fid)
            for key, val in p.items():
                d[key].update(val)

        # restructure provider into new format
        tree = {}
        for key in d.keys():
            for model, params in d[key].items():
                if model not in tree:
                    tree[model] = params
                else:
                    tree[model].update(params)
                model_type = tree[model].pop('type')     
                model_format = tree[model].get('format', None)   
                model_files = tree[model].pop('model_file', None)
                grid_file = tree[model].pop('grid_file', None)
                variable = tree[model].pop('variable', None)  
                scale = tree[model].pop('scale', None)     
                if key == 'elevation':
                    tree[model]['z'] = {}
                    tree[model]['z']['model_file'] = model_files
                    tree[model]['z']['variable'] = variable
                    # add grid file for OTIS and ATLAS formats
                    if grid_file is not None and (model_format != 'TMD3'):
                        tree[model]['z']['grid_file'] = grid_file
                    # assign units based on scale
                    if scale is not None and scale == 0.01:
                        tree[model]['z']['units'] = 'cm'
                    elif scale is not None and scale == 0.001:
                        tree[model]['z']['units'] = 'mm'
                    else:
                        tree[model]['z']['units'] = 'm'
                elif key == 'current':
                    tree[model]['u'] = {}
                    tree[model]['v'] = {}
                    tree[model]['u']['model_file'] = model_files['u']
                    if 'v' in model_files:
                        tree[model]['v']['model_file'] = model_files['v']
                    else:
                        tree[model]['v']['model_file'] = model_files['u']
                    # add grid file for OTIS and ATLAS formats
                    if grid_file is not None and (model_format != 'TMD3'):
                        tree[model]['u']['grid_file'] = grid_file
                        tree[model]['v']['grid_file'] = grid_file
                    # assign long names to variables
                    tree[model]['u']['variable'] = long_name['u']
                    tree[model]['v']['variable'] = long_name['v']
                    # assign units based on scale
                    if scale is not None and scale == 0.0001:
                        tree[model]['u']['units'] = 'cm^2/s'
                        tree[model]['v']['units'] = 'cm^2/s'
                    elif scale is not None and scale == 1.0:
                        tree[model]['u']['units'] = 'cm/s'
                        tree[model]['v']['units'] = 'cm/s'
                    else:
                        tree[model]['u']['units'] = 'm^2/s'
                        tree[model]['v']['units'] = 'm^2/s'
                        
        # writing model parameters back to provider JSON file
        with provider.open('w', encoding='utf-8') as fid:
            indent = 4 if args.pretty else None
            json.dump(tree, fid, indent=indent, sort_keys=True)

if __name__ == '__main__':
    main()