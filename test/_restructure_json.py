import sys
import json

# variable long names for tidal currents
long_name = dict(u='zonal_tidal_current', v='meridional_tidal_current')

with open(sys.argv[1],'r') as fid:
    p = json.load(fid)
    d = p.copy()
    model_type = d.pop('type')
    model_format = d.get('format', None)   
    model_files = d.pop('model_file', None)
    constituents = d.pop('constituents', None)
    grid_file = d.pop('grid_file', None)
    variable = d.pop('variable', None)  
    scale = d.pop('scale', None)  
    d['z'] = {}
    d['z']['model_file'] = model_files
    if grid_file is not None:
        d['z']['grid_file'] = grid_file
    d['z']['variable'] = variable
    if scale is not None and scale == 0.01:
        d['z']['units'] = 'cm'
    elif scale is not None and scale == 0.001:
        d['z']['units'] = 'mm'
    else:
        d['z']['units'] = 'm'

if len(sys.argv) > 2:
    with open(sys.argv[2], 'r') as fid:
        p = json.load(fid)
        model_type = p.pop('type')
        model_format = p.get('format', None)   
        model_files = p.pop('model_file', None)
        grid_file = p.pop('grid_file', None)
        variable = p.pop('variable', None)  
        scale = p.pop('scale', None)  
        for key, val in model_files.items():
            d[key] = {}
            d[key]['model_file'] = val
            if grid_file is not None:
                d[key]['grid_file'] = grid_file
            # assign long names to variables
            d[key]['variable'] = long_name[key]
            # assign units based on scale
            if scale is not None and scale == 0.0001:
                d[key]['units'] = 'cm^2/s'
            elif scale is not None and scale == 1.0:
                d[key]['units'] = 'cm/s'
            else:
                d[key]['units'] = 'm^2/s'
        
# write restructured dictionary to file
with open(sys.argv[1], 'w') as fid:
    json.dump(d, fid)
