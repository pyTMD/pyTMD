#!/usr/bin/env python
u"""
tools.py
Written by Tyler Sutterley (08/2025)
Jupyter notebook, user interface and plotting tools

PYTHON DEPENDENCIES:
    numpy: Scientific Computing Tools For Python
        https://numpy.org
        https://numpy.org/doc/stable/user/numpy-for-matlab-users.html
    ipywidgets: interactive HTML widgets for Jupyter notebooks and IPython
        https://ipywidgets.readthedocs.io/en/latest/
    ipyleaflet: Jupyter / Leaflet bridge enabling interactive maps
        https://github.com/jupyter-widgets/ipyleaflet

UPDATE HISTORY:
    Updated 08/2025: use numpy degree to radian conversions
    Updated 07/2025: add a default directory for tide models
        add full screen control to ipyleaflet maps
    Updated 09/2024: removed widget for ATLAS following database update
        added widget for setting constituent to plot in a cotidal chart
    Updated 07/2024: renamed format for netcdf to ATLAS-netcdf
    Updated 04/2024: use wrapper to importlib for optional dependencies
    Updated 12/2023: pass through VBox and HBox
    Updated 08/2023: place matplotlib within try/except statements
    Updated 05/2023: don't set a default directory for tide models
    Updated 04/2023: using pathlib to define and expand paths
    Updated 01/2023: use debug level logging instead of import warnings
    Updated 11/2022: place more imports within try/except statements
    Updated 08/2022: place some imports behind try/except statements
    Updated 05/2022: include world copy jump in webmercator maps
    Updated 03/2022: add marker relocation routines from notebooks
    Updated 02/2022: add leaflet map projections
    Written 09/2021
"""
import copy
import datetime
import numpy as np
import pyTMD.io.model
import pyTMD.utilities

# attempt imports
IPython = pyTMD.utilities.import_dependency('IPython')
ipyleaflet = pyTMD.utilities.import_dependency('ipyleaflet')
ipywidgets = pyTMD.utilities.import_dependency('ipywidgets')
pyproj = pyTMD.utilities.import_dependency('pyproj')

class widgets:
    def __init__(self, **kwargs):
        # set default keyword options
        kwargs.setdefault('style', {})
        # set style
        self.style = copy.copy(kwargs['style'])
        # pass through some ipywidgets objects
        self.HBox = ipywidgets.HBox
        self.VBox = ipywidgets.VBox

        # default working data directory for tide models
        default_directory = pyTMD.utilities.compressuser(
            pyTMD.utilities.get_data_path('data')
        )
        # set the directory with tide models
        self.directory = ipywidgets.Text(
            value=str(default_directory),
            description='Directory:',
            disabled=False
        )

        # dropdown menu for setting tide model
        model_list = sorted(pyTMD.io.model.ocean_elevation() +
            pyTMD.io.model.load_elevation())
        self.model = ipywidgets.Dropdown(
            options=model_list,
            value='GOT4.10',
            description='Model:',
            disabled=False,
            style=self.style,
        )

        # dropdown menu for setting model constituents
        constituents_list = [
            'q1','o1','p1','k1','n2','m2','s2','k2','s1','m4'
        ]
        self.constituents = ipywidgets.Dropdown(
            options=constituents_list,
            value='m2',
            description='Constituents:',
            disabled=False,
            style=self.style,
        )

        # checkbox for setting if tide files are compressed
        self.compress = ipywidgets.Checkbox(
            value=False,
            description='Compressed?',
            disabled=False,
            style=self.style,
        )

        # date picker widget for setting time
        self.datepick = ipywidgets.DatePicker(
            description='Date:',
            value = datetime.date.today(),
            disabled=False,
            style=self.style,
        )

# draw ipyleaflet map
class leaflet:
    def __init__(self, projection='Global', **kwargs):
        # set default keyword arguments
        kwargs.setdefault('map',None)
        kwargs.setdefault('attribution',True)
        kwargs.setdefault('full_screen_control', False)
        kwargs.setdefault('zoom',1)
        kwargs.setdefault('zoom_control',False)
        kwargs.setdefault('scale_control',False)
        kwargs.setdefault('cursor_control',True)
        kwargs.setdefault('layer_control',True)
        kwargs.setdefault('center',(39,-108))
        # create basemap in projection
        if (projection == 'Global'):
            self.map = ipyleaflet.Map(center=kwargs['center'],
                zoom=kwargs['zoom'], max_zoom=15, world_copy_jump=True,
                attribution_control=kwargs['attribution'],
                basemap=ipyleaflet.basemaps.Esri.WorldTopoMap)
            self.crs = 'EPSG:3857'
        elif (projection == 'North'):
            self.map = ipyleaflet.Map(center=kwargs['center'],
                zoom=kwargs['zoom'], max_zoom=24,
                attribution_control=kwargs['attribution'],
                basemap=ipyleaflet.basemaps.Esri.ArcticOceanBase,
                crs=ipyleaflet.projections.EPSG5936.ESRIBasemap)
            self.map.add(ipyleaflet.basemaps.Esri.ArcticOceanReference)
            self.crs = 'EPSG:5936'
        elif (projection == 'South'):
            self.map = ipyleaflet.Map(center=kwargs['center'],
                zoom=kwargs['zoom'], max_zoom=9,
                attribution_control=kwargs['attribution'],
                basemap=ipyleaflet.basemaps.Esri.AntarcticBasemap,
                crs=ipyleaflet.projections.EPSG3031.ESRIBasemap)
            self.crs = 'EPSG:3031'
        else:
            # use a predefined ipyleaflet map
            self.map = kwargs['map']
            self.crs = self.map.crs['name']
        # add control for full screen
        if kwargs['full_screen_control']:
            self.full_screen_control = ipyleaflet.FullScreenControl()
            self.map.add(self.full_screen_control)
        # add control for layers
        if kwargs['layer_control']:
            self.layer_control = ipyleaflet.LayersControl(position='topleft')
            self.map.add(self.layer_control)
            self.layers = self.map.layers
        # add control for zoom
        if kwargs['zoom_control']:
            zoom_slider = ipywidgets.IntSlider(description='Zoom level:',
                min=self.map.min_zoom, max=self.map.max_zoom, value=self.map.zoom)
            ipywidgets.jslink((zoom_slider, 'value'), (self.map, 'zoom'))
            zoom_control = ipyleaflet.WidgetControl(widget=zoom_slider,
                position='topright')
            self.map.add(zoom_control)
        # add control for spatial scale bar
        if kwargs['scale_control']:
            scale_control = ipyleaflet.ScaleControl(position='topright')
            self.map.add(scale_control)
        # add control for cursor position
        if kwargs['cursor_control']:
            self.cursor = ipywidgets.Label()
            cursor_control = ipyleaflet.WidgetControl(widget=self.cursor,
                position='bottomleft')
            self.map.add(cursor_control)
            # keep track of cursor position
            self.map.on_interaction(self.handle_interaction)
        # add control for marker
        if kwargs['marker_control']:
            # add marker with default location
            self.marker = ipyleaflet.Marker(location=kwargs['center'],
                draggable=True)
            self.map.add(self.marker)
            # add text with marker location
            self.marker_text = ipywidgets.Text(
                value='{0:0.8f},{1:0.8f}'.format(*kwargs['center']),
                description='Lat/Lon:',
                disabled=False)
            # watch marker widgets for changes
            self.marker.observe(self.set_marker_text)
            self.marker_text.observe(self.set_marker_location)
            self.map.observe(self.set_map_center)
            # add control for marker location
            marker_control = ipyleaflet.WidgetControl(
                widget=self.marker_text, position='bottomright')
            self.map.add(marker_control)

    # convert points to EPSG:4326
    def transform(self, x, y, proj4def):
        # convert geolocation variable to EPSG:4326
        crs1 = pyproj.CRS.from_string(proj4def)
        crs2 = pyproj.CRS.from_string('EPSG:4326')
        trans = pyproj.Transformer.from_crs(crs1, crs2, always_xy=True)
        return trans.transform(x, y)

    # fix longitudes to be -180:180
    def wrap_longitudes(self, lon):
        phi = np.arctan2(np.sin(np.radians(lon)), np.cos(np.radians(lon)))
        # convert phi from radians to degrees
        return np.degrees(phi)

    # add function for setting marker text if location changed
    def set_marker_text(self, sender):
        LAT,LON = self.marker.location
        self.marker_text.value = '{0:0.8f},{1:0.8f}'.format(LAT,
            self.wrap_longitudes(LON))

    # add function for setting map center if location changed
    def set_map_center(self, sender):
        self.map.center = self.marker.location

    # add function for setting marker location if text changed
    def set_marker_location(self, sender):
        LAT,LON = [float(i) for i in self.marker_text.value.split(',')]
        self.marker.location = (LAT,LON)

    # handle cursor movements for label
    def handle_interaction(self, **kwargs):
        if (kwargs.get('type') == 'mousemove'):
            lat,lon = kwargs.get('coordinates')
            lon = self.wrap_longitudes(lon)
            self.cursor.value = u"""Latitude: {d[0]:8.4f}\u00B0,
                Longitude: {d[1]:8.4f}\u00B0""".format(d=[lat,lon])
