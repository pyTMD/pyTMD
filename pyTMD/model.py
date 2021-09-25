#!/usr/bin/env python
u"""
model.py
Written by Tyler Sutterley (09/2021)
Class with parameters for named tide models

UPDATE HISTORY:
    Written 09/2021
"""
import os
import re
import copy

class model:
    def __init__(self, directory, **kwargs):
        # set default keyword arguments
        kwargs.setdefault('compressed',False)
        kwargs.setdefault('format','netcdf')
        self.atl03 = None
        self.atl06 = None
        self.atl07 = None
        self.atl11 = None
        self.atl12 = None
        self.compressed = copy.copy(kwargs['compressed'])
        self.constituents = None
        self.description = None
        self.directory = os.path.expanduser(directory)
        self.format = copy.copy(kwargs['format'])
        self.gla12 = None
        self.grid_file = None
        self.long_name = None
        self.model_file = None
        self.name = None
        self.projection = None
        self.scale = 1.0
        self.type = None

    def grid(self,m):
        """Create a model object from known tide grid files
        """
        # model name
        self.name = m
        # compression and output flags
        gzip = '.gz' if self.compressed else ''
        suffix = '.nc' if (self.format == 'netcdf') else ''
        # select between known tide models
        if (m == 'CATS0201'):
            self.grid_file = os.path.join(self.directory,
                'cats0201_tmd','grid_CATS')
        elif (m == 'CATS2008'):
            self.grid_file = os.path.join(self.directory,
                'CATS2008','grid_CATS2008')
        elif (m == 'CATS2008_load'):
            self.grid_file = os.path.join(self.directory,
                'CATS2008a_SPOTL_Load','grid_CATS2008a_opt')
        elif (m == 'TPXO9-atlas'):
            model_directory = os.path.join(self.directory,'TPXO9_atlas')
            self.grid_file = os.path.join(model_directory,
                '{0}{1}{2}'.format('grid_tpxo9_atlas',suffix,gzip))
        elif (m == 'TPXO9-atlas-v2'):
            model_directory = os.path.join(self.directory,'TPXO9_atlas_v2')
            self.grid_file = os.path.join(model_directory,
                '{0}{1}{2}'.format('grid_tpxo9_atlas_30_v2',suffix,gzip))
        elif (m == 'TPXO9-atlas-v3'):
            model_directory = os.path.join(self.directory,'TPXO9_atlas_v3')
            self.grid_file = os.path.join(model_directory,
                '{0}{1}{2}'.format('grid_tpxo9_atlas_30_v3',suffix,gzip))
        elif (m == 'TPXO9-atlas-v4'):
            model_directory = os.path.join(self.directory,'TPXO9_atlas_v4')
            self.grid_file = os.path.join(model_directory,
                '{0}{1}{2}'.format('grid_tpxo9_atlas_30_v4',suffix,gzip))
        elif (m == 'TPXO9.1'):
            self.grid_file = os.path.join(self.directory,
                'TPXO9.1','DATA','grid_tpxo9')
        elif (m == 'TPXO8-atlas'):
            self.grid_file = os.path.join(self.directory,
                'tpxo8_atlas','grid_tpxo8atlas_30_v1')
        elif (m == 'TPXO7.2'):
            self.grid_file = os.path.join(self.directory,
                'TPXO7.2_tmd','grid_tpxo7.2')
        elif (m == 'TPXO7.2_load'):
            self.grid_file = os.path.join(self.directory,
                'TPXO7.2_load','grid_tpxo6.2')
        elif (m == 'AODTM-5'):
            self.grid_file = os.path.join(self.directory,
                'aodtm5_tmd','grid_Arc5km')
        elif (m == 'AOTIM-5'):
            self.grid_file = os.path.join(self.directory,
                'aotim5_tmd','grid_Arc5km')
        elif (m == 'AOTIM-5-2018'):
            self.grid_file = os.path.join(self.directory,
                'Arc5km2018','grid_Arc5km2018')
        elif (m == 'Gr1km-v2'):
            self.grid_file = os.path.join(self.directory,
                'greenlandTMD_v2','grid_Greenland8.v2')
        else:
            raise Exception("Unlisted tide model")
        # return the model parameters
        return self

    def elevation(self,m):
        """Create a model object from known tidal elevation models
        """
        # model name
        self.name = m
        # model type
        self.type = 'z'
        # compression and output flags
        gzip = '.gz' if self.compressed else ''
        suffix = '.nc' if (self.format == 'netcdf') else ''
        # select between known tide models
        if (m == 'CATS0201'):
            self.grid_file = os.path.join(self.directory,
                'cats0201_tmd','grid_CATS')
            self.model_file = os.path.join(self.directory,
                'cats0201_tmd','h0_CATS02_01')
            self.format = 'OTIS'
            self.projection = '4326'
            # model description and references
            self.reference = ('https://mail.esr.org/polar_tide_models/'
                'Model_CATS0201.html')
            self.atl03 = 'tide_ocean'
            self.atl06 = 'tide_ocean'
            self.atl07 = 'height_segment_ocean'
            self.atl11 = 'tide_ocean'
            self.atl12 = 'tide_ocean_seg'
            self.gla12 = 'd_ocElv'
            self.variable = 'tide_ocean'
            self.long_name = "Ocean Tide"
            self.description = ("Ocean Tides including diurnal and "
                "semi-diurnal (harmonic analysis), and longer period "
                "tides (dynamic and self-consistent equilibrium).")
        elif (m == 'CATS2008'):
            self.grid_file = os.path.join(self.directory,
                'CATS2008','grid_CATS2008')
            self.model_file = os.path.join(self.directory,
                'CATS2008','hf.CATS2008.out')
            self.format = 'OTIS'
            self.projection = 'CATS2008'
            # model description and references
            self.reference = ('https://www.esr.org/research/'
                'polar-tide-models/list-of-polar-tide-models/cats2008/')
            self.atl03 = 'tide_ocean'
            self.atl06 = 'tide_ocean'
            self.atl07 = 'height_segment_ocean'
            self.atl11 = 'tide_ocean'
            self.atl12 = 'tide_ocean_seg'
            self.gla12 = 'd_ocElv'
            self.variable = 'tide_ocean'
            self.long_name = "Ocean Tide"
            self.description = ("Ocean Tides including diurnal and "
                "semi-diurnal (harmonic analysis), and longer period "
                "tides (dynamic and self-consistent equilibrium).")
        elif (m == 'CATS2008_load'):
            self.grid_file = os.path.join(self.directory,
                'CATS2008a_SPOTL_Load','grid_CATS2008a_opt')
            self.model_file = os.path.join(self.directory,
                'CATS2008a_SPOTL_Load','h_CATS2008a_SPOTL_load')
            self.format = 'OTIS'
            self.projection = 'CATS2008'
            # model description and references
            self.reference = ('https://www.esr.org/research/'
                'polar-tide-models/list-of-polar-tide-models/cats2008/')
            self.atl03 = 'tide_load'
            self.atl06 = 'tide_load'
            self.atl07 = 'height_segment_load'
            self.atl11 = 'tide_load'
            self.atl12 = 'tide_load_seg'
            self.gla12 = 'd_ldElv'
            self.variable = 'tide_load'
            self.long_name = "Load Tide"
            self.description = ("Local displacement due to Ocean "
                "Loading (-6 to 0 cm)")
        elif (m == 'TPXO9-atlas'):
            model_directory = os.path.join(self.directory,'TPXO9_atlas')
            self.grid_file = os.path.join(model_directory,
                '{0}{1}{2}'.format('grid_tpxo9_atlas',suffix,gzip))
            model_files = ['h_q1_tpxo9_atlas_30','h_o1_tpxo9_atlas_30',
                'h_p1_tpxo9_atlas_30','h_k1_tpxo9_atlas_30',
                'h_n2_tpxo9_atlas_30','h_m2_tpxo9_atlas_30',
                'h_s2_tpxo9_atlas_30','h_k2_tpxo9_atlas_30',
                'h_m4_tpxo9_atlas_30','h_ms4_tpxo9_atlas_30',
                'h_mn4_tpxo9_atlas_30','h_2n2_tpxo9_atlas_30']
            self.model_file = [os.path.join(model_directory,
                '{0}{1}{2}'.format(f,suffix,gzip)) for f in model_files]
            self.projection = '4326'
            self.scale = 1.0/1000.0
            # model description and references
            self.reference = ('http://volkov.oce.orst.edu/tides/'
                'tpxo9_atlas.html')
            self.atl03 = 'tide_ocean'
            self.atl06 = 'tide_ocean'
            self.atl07 = 'height_segment_ocean'
            self.atl11 = 'tide_ocean'
            self.atl12 = 'tide_ocean_seg'
            self.gla12 = 'd_ocElv'
            self.variable = 'tide_ocean'
            self.long_name = "Ocean Tide"
            self.description = ("Ocean Tides including diurnal and "
                "semi-diurnal (harmonic analysis), and longer period "
                "tides (dynamic and self-consistent equilibrium).")
        elif (m == 'TPXO9-atlas-v2'):
            model_directory = os.path.join(self.directory,'TPXO9_atlas_v2')
            self.grid_file = os.path.join(model_directory,
                '{0}{1}{2}'.format('grid_tpxo9_atlas_30_v2',suffix,gzip))
            model_files = ['h_q1_tpxo9_atlas_30_v2','h_o1_tpxo9_atlas_30_v2',
                'h_p1_tpxo9_atlas_30_v2','h_k1_tpxo9_atlas_30_v2',
                'h_n2_tpxo9_atlas_30_v2','h_m2_tpxo9_atlas_30_v2',
                'h_s2_tpxo9_atlas_30_v2','h_k2_tpxo9_atlas_30_v2',
                'h_m4_tpxo9_atlas_30_v2','h_ms4_tpxo9_atlas_30_v2',
                'h_mn4_tpxo9_atlas_30_v2','h_2n2_tpxo9_atlas_30_v2']
            self.model_file = [os.path.join(model_directory,
                '{0}{1}{2}'.format(f,suffix,gzip)) for f in model_files]
            self.projection = '4326'
            self.scale = 1.0/1000.0
            # model description and references
            self.reference = 'https://www.tpxo.net/global/tpxo9-atlas'
            self.atl03 = 'tide_ocean'
            self.atl06 = 'tide_ocean'
            self.atl07 = 'height_segment_ocean'
            self.atl11 = 'tide_ocean'
            self.atl12 = 'tide_ocean_seg'
            self.gla12 = 'd_ocElv'
            self.variable = 'tide_ocean'
            self.long_name = "Ocean Tide"
            self.description = ("Ocean Tides including diurnal and "
                "semi-diurnal (harmonic analysis), and longer period "
                "tides (dynamic and self-consistent equilibrium).")
        elif (m == 'TPXO9-atlas-v3'):
            model_directory = os.path.join(self.directory,'TPXO9_atlas_v3')
            self.grid_file = os.path.join(model_directory,
                '{0}{1}{2}'.format('grid_tpxo9_atlas_30_v3',suffix,gzip))
            model_files = ['h_q1_tpxo9_atlas_30_v3','h_o1_tpxo9_atlas_30_v3',
                'h_p1_tpxo9_atlas_30_v3','h_k1_tpxo9_atlas_30_v3',
                'h_n2_tpxo9_atlas_30_v3','h_m2_tpxo9_atlas_30_v3',
                'h_s2_tpxo9_atlas_30_v3','h_k2_tpxo9_atlas_30_v3',
                'h_m4_tpxo9_atlas_30_v3','h_ms4_tpxo9_atlas_30_v3',
                'h_mn4_tpxo9_atlas_30_v3','h_2n2_tpxo9_atlas_30_v3',
                'h_mf_tpxo9_atlas_30_v3','h_mm_tpxo9_atlas_30_v3']
            self.model_file = [os.path.join(model_directory,
                '{0}{1}{2}'.format(f,suffix,gzip)) for f in model_files]
            self.projection = '4326'
            self.scale = 1.0/1000.0
            # model description and references
            self.reference = 'https://www.tpxo.net/global/tpxo9-atlas'
            self.atl03 = 'tide_ocean'
            self.atl06 = 'tide_ocean'
            self.atl07 = 'height_segment_ocean'
            self.atl11 = 'tide_ocean'
            self.atl12 = 'tide_ocean_seg'
            self.gla12 = 'd_ocElv'
            self.variable = 'tide_ocean'
            self.long_name = "Ocean Tide"
            self.description = ("Ocean Tides including diurnal and "
                "semi-diurnal (harmonic analysis), and longer period "
                "tides (dynamic and self-consistent equilibrium).")
        elif (m == 'TPXO9-atlas-v4'):
            model_directory = os.path.join(self.directory,'TPXO9_atlas_v4')
            self.grid_file = os.path.join(model_directory,
                '{0}{1}{2}'.format('grid_tpxo9_atlas_30_v4',suffix,gzip))
            model_files = ['h_q1_tpxo9_atlas_30_v4','h_o1_tpxo9_atlas_30_v4',
                'h_p1_tpxo9_atlas_30_v4','h_k1_tpxo9_atlas_30_v4',
                'h_n2_tpxo9_atlas_30_v4','h_m2_tpxo9_atlas_30_v4',
                'h_s2_tpxo9_atlas_30_v4','h_k2_tpxo9_atlas_30_v4',
                'h_m4_tpxo9_atlas_30_v4','h_ms4_tpxo9_atlas_30_v4',
                'h_mn4_tpxo9_atlas_30_v4','h_2n2_tpxo9_atlas_30_v4',
                'h_mf_tpxo9_atlas_30_v4','h_mm_tpxo9_atlas_30_v4']
            self.model_file = [os.path.join(model_directory,
                '{0}{1}{2}'.format(f,suffix,gzip)) for f in model_files]
            self.projection = '4326'
            self.scale = 1.0/1000.0
            # model description and references
            self.reference = 'https://www.tpxo.net/global/tpxo9-atlas'
            self.atl03 = 'tide_ocean'
            self.atl06 = 'tide_ocean'
            self.atl07 = 'height_segment_ocean'
            self.atl11 = 'tide_ocean'
            self.atl12 = 'tide_ocean_seg'
            self.gla12 = 'd_ocElv'
            self.variable = 'tide_ocean'
            self.long_name = "Ocean Tide"
            self.description = ("Ocean Tides including diurnal and "
                "semi-diurnal (harmonic analysis), and longer period "
                "tides (dynamic and self-consistent equilibrium).")
        elif (m == 'TPXO9.1'):
            self.grid_file = os.path.join(self.directory,
                'TPXO9.1','DATA','grid_tpxo9')
            self.model_file = os.path.join(self.directory,
                'TPXO9.1','DATA','h_tpxo9.v1')
            self.format = 'OTIS'
            self.projection = '4326'
            # model description and references
            self.reference = ('http://volkov.oce.orst.edu/'
                'tides/global.html')
            self.atl03 = 'tide_ocean'
            self.atl06 = 'tide_ocean'
            self.atl07 = 'height_segment_ocean'
            self.atl11 = 'tide_ocean'
            self.atl12 = 'tide_ocean_seg'
            self.gla12 = 'd_ocElv'
            self.variable = 'tide_ocean'
            self.long_name = "Ocean Tide"
            self.description = ("Ocean Tides including diurnal and "
                "semi-diurnal (harmonic analysis), and longer period "
                "tides (dynamic and self-consistent equilibrium).")
        elif (m == 'TPXO8-atlas'):
            self.grid_file = os.path.join(self.directory,
                'tpxo8_atlas','grid_tpxo8atlas_30_v1')
            self.model_file = os.path.join(self.directory,
                'tpxo8_atlas','hf.tpxo8_atlas_30_v1')
            self.format = 'ATLAS'
            self.projection = '4326'
            # model description and references
            self.reference = ('http://volkov.oce.orst.edu/'
                'tides/tpxo8_atlas.html')
            self.atl03 = 'tide_ocean'
            self.atl06 = 'tide_ocean'
            self.atl07 = 'height_segment_ocean'
            self.atl11 = 'tide_ocean'
            self.atl12 = 'tide_ocean_seg'
            self.gla12 = 'd_ocElv'
            self.variable = 'tide_ocean'
            self.long_name = "Ocean Tide"
            self.description = ("Ocean Tides including diurnal and "
                "semi-diurnal (harmonic analysis), and longer period "
                "tides (dynamic and self-consistent equilibrium).")
        elif (m == 'TPXO7.2'):
            self.grid_file = os.path.join(self.directory,
                'TPXO7.2_tmd','grid_tpxo7.2')
            self.model_file = os.path.join(self.directory,
                'TPXO7.2_tmd','h_tpxo7.2')
            self.format = 'OTIS'
            self.projection = '4326'
            # model description and references
            self.reference = ('http://volkov.oce.orst.edu/'
                'tides/global.html')
            self.atl03 = 'tide_ocean'
            self.atl06 = 'tide_ocean'
            self.atl07 = 'height_segment_ocean'
            self.atl11 = 'tide_ocean'
            self.atl12 = 'tide_ocean_seg'
            self.gla12 = 'd_ocElv'
            self.variable = 'tide_ocean'
            self.long_name = "Ocean Tide"
            self.description = ("Ocean Tides including diurnal and "
                "semi-diurnal (harmonic analysis), and longer period "
                "tides (dynamic and self-consistent equilibrium).")
        elif (m == 'TPXO7.2_load'):
            self.grid_file = os.path.join(self.directory,
                'TPXO7.2_load','grid_tpxo6.2')
            self.model_file = os.path.join(self.directory,
                'TPXO7.2_load','h_tpxo7.2_load')
            self.format = 'OTIS'
            self.projection = '4326'
            # model description and references
            self.reference = ('http://volkov.oce.orst.edu/'
                'tides/global.html')
            self.atl03 = 'tide_load'
            self.atl06 = 'tide_load'
            self.atl07 = 'height_segment_load'
            self.atl11 = 'tide_load'
            self.atl12 = 'tide_load_seg'
            self.gla12 = 'd_ldElv'
            self.variable = 'tide_load'
            self.long_name = "Load Tide"
            self.description = ("Local displacement due to Ocean "
                "Loading (-6 to 0 cm)")
        elif (m == 'AODTM-5'):
            self.grid_file = os.path.join(self.directory,
                'aodtm5_tmd','grid_Arc5km')
            self.model_file = os.path.join(self.directory,
                'aodtm5_tmd','h0_Arc5km.oce')
            self.format = 'OTIS'
            self.projection = 'PSNorth'
            # model description and references
            self.reference = ('https://www.esr.org/research/'
                'polar-tide-models/list-of-polar-tide-models/'
                'aodtm-5/')
            self.atl03 = 'tide_ocean'
            self.atl06 = 'tide_ocean'
            self.atl07 = 'height_segment_ocean'
            self.atl11 = 'tide_ocean'
            self.atl12 = 'tide_ocean_seg'
            self.gla12 = 'd_ocElv'
            self.variable = 'tide_ocean'
            self.long_name = "Ocean Tide"
            self.description = ("Ocean Tides including diurnal and "
                "semi-diurnal (harmonic analysis), and longer period "
                "tides (dynamic and self-consistent equilibrium).")
        elif (m == 'AOTIM-5'):
            self.grid_file = os.path.join(self.directory,
                'aotim5_tmd','grid_Arc5km')
            self.model_file = os.path.join(self.directory,
                'aotim5_tmd','h_Arc5km.oce')
            self.format = 'OTIS'
            self.projection = 'PSNorth'
            # model description and references
            self.reference = ('https://www.esr.org/research/'
                'polar-tide-models/list-of-polar-tide-models/'
                'aotim-5/')
            self.atl03 = 'tide_ocean'
            self.atl06 = 'tide_ocean'
            self.atl07 = 'height_segment_ocean'
            self.atl11 = 'tide_ocean'
            self.atl12 = 'tide_ocean_seg'
            self.gla12 = 'd_ocElv'
            self.variable = 'tide_ocean'
            self.long_name = "Ocean Tide"
            self.description = ("Ocean Tides including diurnal and "
                "semi-diurnal (harmonic analysis), and longer period "
                "tides (dynamic and self-consistent equilibrium).")
        elif (m == 'AOTIM-5-2018'):
            self.grid_file = os.path.join(self.directory,
                'Arc5km2018','grid_Arc5km2018')
            self.model_file = os.path.join(self.directory,
                'Arc5km2018','h_Arc5km2018')
            self.format = 'OTIS'
            self.projection = 'PSNorth'
            # model description and references
            self.reference = ('https://www.esr.org/research/'
                'polar-tide-models/list-of-polar-tide-models/'
                'aotim-5/')
            self.atl03 = 'tide_ocean'
            self.atl06 = 'tide_ocean'
            self.atl07 = 'height_segment_ocean'
            self.atl11 = 'tide_ocean'
            self.atl12 = 'tide_ocean_seg'
            self.gla12 = 'd_ocElv'
            self.variable = 'tide_ocean'
            self.long_name = "Ocean Tide"
            self.description = ("Ocean Tides including diurnal and "
                "semi-diurnal (harmonic analysis), and longer period "
                "tides (dynamic and self-consistent equilibrium).")
        elif (m == 'Gr1km-v2'):
            self.grid_file = os.path.join(self.directory,
                'greenlandTMD_v2','grid_Greenland8.v2')
            self.model_file = os.path.join(self.directory,
                'greenlandTMD_v2','h_Greenland8.v2')
            self.format = 'OTIS'
            self.projection = '3413'
            # model description and references
            self.reference = 'https://doi.org/10.1002/2016RG000546'
            self.atl03 = 'tide_ocean'
            self.atl06 = 'tide_ocean'
            self.atl07 = 'height_segment_ocean'
            self.atl11 = 'tide_ocean'
            self.atl12 = 'tide_ocean_seg'
            self.gla12 = 'd_ocElv'
            self.variable = 'tide_ocean'
            self.long_name = "Ocean Tide"
            self.description = ("Ocean Tides including diurnal and "
                "semi-diurnal (harmonic analysis), and longer period "
                "tides (dynamic and self-consistent equilibrium).")
        elif (m == 'GOT4.7'):
            model_directory = os.path.join(self.directory,
                'GOT4.7','grids_oceantide')
            model_files = ['q1.d','o1.d','p1.d','k1.d','n2.d',
                'm2.d','s2.d','k2.d','s1.d','m4.d']
            self.model_file = [os.path.join(model_directory,
                '{0}{1}'.format(f,gzip)) for f in model_files]
            self.format = 'GOT'
            self.scale = 1.0/100.0
            # model description and references
            self.reference = ('https://denali.gsfc.nasa.gov/'
                'personal_pages/ray/MiscPubs/'
                '19990089548_1999150788.pdf')
            self.atl03 = 'tide_ocean'
            self.atl06 = 'tide_ocean'
            self.atl07 = 'height_segment_ocean'
            self.atl11 = 'tide_ocean'
            self.atl12 = 'tide_ocean_seg'
            self.gla12 = 'd_ocElv'
            self.variable = 'tide_ocean'
            self.long_name = "Ocean Tide"
            self.description = ("Ocean Tides including diurnal and "
                "semi-diurnal (harmonic analysis), and longer period "
                "tides (dynamic and self-consistent equilibrium).")
        elif (m == 'GOT4.7_load'):
            model_directory = os.path.join(self.directory,
                'GOT4.7','grids_loadtide')
            model_files = ['q1load.d','o1load.d',
                'p1load.d','k1load.d','n2load.d',
                'm2load.d','s2load.d','k2load.d',
                's1load.d','m4load.d']
            self.model_file = [os.path.join(model_directory,
                '{0}{1}'.format(f,gzip)) for f in model_files]
            self.format = 'GOT'
            self.scale = 1.0/1000.0
            # model description and references
            self.reference = ('https://denali.gsfc.nasa.gov/'
                'personal_pages/ray/MiscPubs/'
                '19990089548_1999150788.pdf')
            self.atl03 = 'tide_load'
            self.atl06 = 'tide_load'
            self.atl07 = 'height_segment_load'
            self.atl11 = 'tide_load'
            self.atl12 = 'tide_load_seg'
            self.gla12 = 'd_ldElv'
            self.variable = 'tide_load'
            self.long_name = "Load Tide"
            self.description = ("Local displacement due to Ocean "
                "Loading (-6 to 0 cm)")
        elif (m == 'GOT4.8'):
            model_directory = os.path.join(self.directory,
                'got4.8','grids_oceantide')
            model_files = ['q1.d','o1.d','p1.d','k1.d','n2.d',
                'm2.d','s2.d','k2.d','s1.d','m4.d']
            self.model_file = [os.path.join(model_directory,
                '{0}{1}'.format(f,gzip)) for f in model_files]
            self.format = 'GOT'
            self.scale = 1.0/100.0
            # model description and references
            self.reference = ('https://denali.gsfc.nasa.gov/'
                'personal_pages/ray/MiscPubs/'
                '19990089548_1999150788.pdf')
            self.atl03 = 'tide_ocean'
            self.atl06 = 'tide_ocean'
            self.atl07 = 'height_segment_ocean'
            self.atl11 = 'tide_ocean'
            self.atl12 = 'tide_ocean_seg'
            self.gla12 = 'd_ocElv'
            self.variable = 'tide_ocean'
            self.long_name = "Ocean Tide"
            self.description = ("Ocean Tides including diurnal and "
                "semi-diurnal (harmonic analysis), and longer period "
                "tides (dynamic and self-consistent equilibrium).")
        elif (m == 'GOT4.8_load'):
            model_directory = os.path.join(self.directory,
                'got4.8','grids_loadtide')
            model_files = ['q1load.d','o1load.d',
                'p1load.d','k1load.d','n2load.d',
                'm2load.d','s2load.d','k2load.d',
                's1load.d','m4load.d']
            self.model_file = [os.path.join(model_directory,
                '{0}{1}'.format(f,gzip)) for f in model_files]
            self.format = 'GOT'
            self.scale = 1.0/1000.0
            # model description and references
            self.reference = ('https://denali.gsfc.nasa.gov/'
                'personal_pages/ray/MiscPubs/'
                '19990089548_1999150788.pdf')
            self.atl03 = 'tide_load'
            self.atl06 = 'tide_load'
            self.atl07 = 'height_segment_load'
            self.atl11 = 'tide_load'
            self.atl12 = 'tide_load_seg'
            self.gla12 = 'd_ldElv'
            self.variable = 'tide_load'
            self.long_name = "Load Tide"
            self.description = ("Local displacement due to Ocean "
                "Loading (-6 to 0 cm)")
        elif (m == 'GOT4.10'):
            model_directory = os.path.join(self.directory,
                'GOT4.10c','grids_oceantide')
            model_files = ['q1.d','o1.d','p1.d','k1.d','n2.d',
                'm2.d','s2.d','k2.d','s1.d','m4.d']
            self.model_file = [os.path.join(model_directory,
                '{0}{1}'.format(f,gzip)) for f in model_files]
            self.format = 'GOT'
            self.scale = 1.0/100.0
            # model description and references
            self.reference = ('https://denali.gsfc.nasa.gov/'
                'personal_pages/ray/MiscPubs/'
                '19990089548_1999150788.pdf')
            self.atl03 = 'tide_ocean'
            self.atl06 = 'tide_ocean'
            self.atl07 = 'height_segment_ocean'
            self.atl11 = 'tide_ocean'
            self.atl12 = 'tide_ocean_seg'
            self.gla12 = 'd_ocElv'
            self.variable = 'tide_ocean'
            self.long_name = "Ocean Tide"
            self.description = ("Ocean Tides including diurnal and "
                "semi-diurnal (harmonic analysis), and longer period "
                "tides (dynamic and self-consistent equilibrium).")
        elif (m == 'GOT4.10_load'):
            model_directory = os.path.join(self.directory,
                'GOT4.10c','grids_loadtide')
            model_files = ['q1load.d','o1load.d',
                'p1load.d','k1load.d','n2load.d',
                'm2load.d','s2load.d','k2load.d',
                's1load.d','m4load.d']
            self.model_file = [os.path.join(model_directory,
                '{0}{1}'.format(f,gzip)) for f in model_files]
            self.format = 'GOT'
            self.scale = 1.0/1000.0
            # model description and references
            self.reference = ('https://denali.gsfc.nasa.gov/'
                'personal_pages/ray/MiscPubs/'
                '19990089548_1999150788.pdf')
            self.atl03 = 'tide_load'
            self.atl06 = 'tide_load'
            self.atl07 = 'height_segment_load'
            self.atl11 = 'tide_load'
            self.atl12 = 'tide_load_seg'
            self.gla12 = 'd_ldElv'
            self.variable = 'tide_load'
            self.long_name = "Load Tide"
            self.description = ("Local displacement due to Ocean "
                "Loading (-6 to 0 cm)")
        elif (m == 'FES2014'):
            model_directory = os.path.join(self.directory,
                'fes2014','ocean_tide')
            model_files = ['2n2.nc','eps2.nc','j1.nc','k1.nc',
                'k2.nc','l2.nc','la2.nc','m2.nc','m3.nc','m4.nc',
                'm6.nc','m8.nc','mf.nc','mks2.nc','mm.nc',
                'mn4.nc','ms4.nc','msf.nc','msqm.nc','mtm.nc',
                'mu2.nc','n2.nc','n4.nc','nu2.nc','o1.nc','p1.nc',
                'q1.nc','r2.nc','s1.nc','s2.nc','s4.nc','sa.nc',
                'ssa.nc','t2.nc']
            self.model_file = [os.path.join(model_directory,
                '{0}{1}'.format(f,gzip)) for f in model_files]
            self.constituents = ['2n2','eps2','j1','k1','k2','l2',
                'lambda2','m2','m3','m4','m6','m8','mf','mks2','mm',
                'mn4','ms4','msf','msqm','mtm','mu2','n2','n4','nu2',
                'o1','p1','q1','r2','s1','s2','s4','sa','ssa','t2']
            self.format = 'FES'
            self.scale = 1.0/100.0
            # model description and references
            self.reference = ('https://www.aviso.altimetry.fr/'
                'en/data/products/auxiliary-products/'
                'global-tide-fes.html')
            self.atl03 = 'tide_ocean'
            self.atl06 = 'tide_ocean'
            self.atl07 = 'height_segment_ocean'
            self.atl11 = 'tide_ocean'
            self.atl12 = 'tide_ocean_seg'
            self.gla12 = 'd_ocElv'
            self.variable = 'tide_ocean'
            self.long_name = "Ocean Tide"
            self.description = ("Ocean Tides including diurnal and "
                "semi-diurnal (harmonic analysis), and longer period "
                "tides (dynamic and self-consistent equilibrium).")
        elif (m == 'FES2014_load'):
            model_directory = os.path.join(self.directory,
                'fes2014','load_tide')
            model_files = ['2n2.nc','eps2.nc','j1.nc','k1.nc',
                'k2.nc','l2.nc','la2.nc','m2.nc','m3.nc','m4.nc',
                'm6.nc','m8.nc','mf.nc','mks2.nc','mm.nc',
                'mn4.nc','ms4.nc','msf.nc','msqm.nc','mtm.nc',
                'mu2.nc','n2.nc','n4.nc','nu2.nc','o1.nc','p1.nc',
                'q1.nc','r2.nc','s1.nc','s2.nc','s4.nc','sa.nc',
                'ssa.nc','t2.nc']
            self.model_file = [os.path.join(model_directory,
                '{0}{1}'.format(f,gzip)) for f in model_files]
            self.constituents = ['2n2','eps2','j1','k1','k2','l2',
                'lambda2','m2','m3','m4','m6','m8','mf','mks2','mm',
                'mn4','ms4','msf','msqm','mtm','mu2','n2','n4','nu2',
                'o1','p1','q1','r2','s1','s2','s4','sa','ssa','t2']
            self.format = 'FES'
            self.scale = 1.0/100.0
            # model description and references
            self.reference = ('https://www.aviso.altimetry.fr/'
                'en/data/products/auxiliary-products/'
                'global-tide-fes.html')
            self.atl03 = 'tide_load'
            self.atl06 = 'tide_load'
            self.atl07 = 'height_segment_load'
            self.atl11 = 'tide_load'
            self.atl12 = 'tide_load_seg'
            self.gla12 = 'd_ldElv'
            self.variable = 'tide_load'
            self.long_name = "Load Tide"
            self.description = ("Local displacement due to Ocean "
                "Loading (-6 to 0 cm)")
        else:
            raise Exception("Unlisted tide model")
        # return the model parameters
        return self

    def current(self,m):
        """Create a model object from known tidal current models
        """
        # model name
        self.name = m
        # model type
        self.type = ['u','v']
        # compression and output flags
        gzip = '.gz' if self.compressed else ''
        suffix = '.nc' if (self.format == 'netcdf') else ''
        # select between tide models
        if (m == 'CATS0201'):
            self.grid_file = os.path.join(self.directory,
                'cats0201_tmd','grid_CATS')
            self.model_file = dict(u=os.path.join(self.directory,
                'cats0201_tmd','UV0_CATS02_01'))
            self.format = 'OTIS'
            self.projection = '4326'
            # model description and references
            self.reference = ('https://mail.esr.org/polar_tide_models/'
                'Model_CATS0201.html')
        elif (m == 'CATS2008'):
            self.grid_file = os.path.join(self.directory,
                'CATS2008','grid_CATS2008')
            self.model_file = dict(u=os.path.join(self.directory,
                'CATS2008','uv.CATS2008.out'))
            self.format = 'OTIS'
            self.projection = 'CATS2008'
        elif (m == 'TPXO9-atlas'):
            model_directory = os.path.join(self.directory,'TPXO9_atlas')
            self.grid_file = os.path.join(model_directory,
                '{0}{1}{2}'.format('grid_tpxo9_atlas',suffix,gzip))
            model_files = {}
            model_files['u'] = ['u_q1_tpxo9_atlas_30','u_o1_tpxo9_atlas_30',
                'u_p1_tpxo9_atlas_30','u_k1_tpxo9_atlas_30',
                'u_n2_tpxo9_atlas_30','u_m2_tpxo9_atlas_30',
                'u_s2_tpxo9_atlas_30','u_k2_tpxo9_atlas_30',
                'u_m4_tpxo9_atlas_30','u_ms4_tpxo9_atlas_30',
                'u_mn4_tpxo9_atlas_30','u_2n2_tpxo9_atlas_30']
            model_files['v'] = ['v_q1_tpxo9_atlas_30','v_o1_tpxo9_atlas_30',
                'v_p1_tpxo9_atlas_30','v_k1_tpxo9_atlas_30',
                'v_n2_tpxo9_atlas_30','v_m2_tpxo9_atlas_30',
                'v_s2_tpxo9_atlas_30','v_k2_tpxo9_atlas_30',
                'v_m4_tpxo9_atlas_30','v_ms4_tpxo9_atlas_30',
                'v_mn4_tpxo9_atlas_30','v_2n2_tpxo9_atlas_30']
            self.model_file = {}
            for key,val in model_files.items():
                self.model_file[key] = [os.path.join(model_directory,
                    '{0}{1}{2}'.format(f,suffix,gzip)) for f in val]
            self.projection = '4326'
            self.scale = 1.0/100.0
            # model description and references
            self.reference = ('http://volkov.oce.orst.edu/tides/'
                'tpxo9_atlas.html')
        elif (m == 'TPXO9-atlas-v2'):
            model_directory = os.path.join(self.directory,'TPXO9_atlas_v2')
            self.grid_file = os.path.join(model_directory,
                '{0}{1}{2}'.format('grid_tpxo9_atlas_30_v2',suffix,gzip))
            model_files = {}
            model_files['u'] = ['u_q1_tpxo9_atlas_30_v2','u_o1_tpxo9_atlas_30_v2',
                'u_p1_tpxo9_atlas_30_v2','u_k1_tpxo9_atlas_30_v2',
                'u_n2_tpxo9_atlas_30_v2','u_m2_tpxo9_atlas_30_v2',
                'u_s2_tpxo9_atlas_30_v2','u_k2_tpxo9_atlas_30_v2',
                'u_m4_tpxo9_atlas_30_v2','u_ms4_tpxo9_atlas_30_v2',
                'u_mn4_tpxo9_atlas_30_v2','u_2n2_tpxo9_atlas_30_v2']
            model_files['v'] = ['v_q1_tpxo9_atlas_30_v2','v_o1_tpxo9_atlas_30_v2',
                'v_p1_tpxo9_atlas_30_v2','v_k1_tpxo9_atlas_30_v2',
                'v_n2_tpxo9_atlas_30_v2','v_m2_tpxo9_atlas_30_v2',
                'v_s2_tpxo9_atlas_30_v2','v_k2_tpxo9_atlas_30_v2',
                'v_m4_tpxo9_atlas_30_v2','v_ms4_tpxo9_atlas_30_v2',
                'v_mn4_tpxo9_atlas_30_v2','v_2n2_tpxo9_atlas_30_v2']
            self.model_file = {}
            for key,val in model_files.items():
                self.model_file[key] = [os.path.join(model_directory,
                    '{0}{1}{2}'.format(f,suffix,gzip)) for f in val]
            self.projection = '4326'
            self.scale = 1.0/100.0
            # model description and references
            self.reference = 'https://www.tpxo.net/global/tpxo9-atlas'
        elif (m == 'TPXO9-atlas-v3'):
            model_directory = os.path.join(self.directory,'TPXO9_atlas_v3')
            self.grid_file = os.path.join(model_directory,
                '{0}{1}{2}'.format('grid_tpxo9_atlas_30_v3',suffix,gzip))
            model_files = {}
            model_files['u'] = ['u_q1_tpxo9_atlas_30_v3','u_o1_tpxo9_atlas_30_v3',
                'u_p1_tpxo9_atlas_30_v3','u_k1_tpxo9_atlas_30_v3',
                'u_n2_tpxo9_atlas_30_v3','u_m2_tpxo9_atlas_30_v3',
                'u_s2_tpxo9_atlas_30_v3','u_k2_tpxo9_atlas_30_v3',
                'u_m4_tpxo9_atlas_30_v3','u_ms4_tpxo9_atlas_30_v3',
                'u_mn4_tpxo9_atlas_30_v3','u_2n2_tpxo9_atlas_30_v3']
            model_files['v'] = ['v_q1_tpxo9_atlas_30_v3','v_o1_tpxo9_atlas_30_v3',
                'v_p1_tpxo9_atlas_30_v3','v_k1_tpxo9_atlas_30_v3',
                'v_n2_tpxo9_atlas_30_v3','v_m2_tpxo9_atlas_30_v3',
                'v_s2_tpxo9_atlas_30_v3','v_k2_tpxo9_atlas_30_v3',
                'v_m4_tpxo9_atlas_30_v3','v_ms4_tpxo9_atlas_30_v3',
                'v_mn4_tpxo9_atlas_30_v3','v_2n2_tpxo9_atlas_30_v3']
            self.model_file = {}
            for key,val in model_files.items():
                self.model_file[key] = [os.path.join(model_directory,
                    '{0}{1}{2}'.format(f,suffix,gzip)) for f in val]
            self.projection = '4326'
            self.scale = 1.0/100.0
            # model description and references
            self.reference = 'https://www.tpxo.net/global/tpxo9-atlas'
        elif (m == 'TPXO9-atlas-v4'):
            model_directory = os.path.join(self.directory,'TPXO9_atlas_v4')
            self.grid_file = os.path.join(model_directory,
                '{0}{1}{2}'.format('grid_tpxo9_atlas_30_v4',suffix,gzip))
            model_files = {}
            model_files['u'] = ['u_q1_tpxo9_atlas_30_v4','u_o1_tpxo9_atlas_30_v4',
                'u_p1_tpxo9_atlas_30_v4','u_k1_tpxo9_atlas_30_v4',
                'u_n2_tpxo9_atlas_30_v4','u_m2_tpxo9_atlas_30_v4',
                'u_s2_tpxo9_atlas_30_v4','u_k2_tpxo9_atlas_30_v4',
                'u_m4_tpxo9_atlas_30_v4','u_ms4_tpxo9_atlas_30_v4',
                'u_mn4_tpxo9_atlas_30_v4','u_2n2_tpxo9_atlas_30_v4']
            model_files['v'] = ['v_q1_tpxo9_atlas_30_v4','v_o1_tpxo9_atlas_30_v4',
                'v_p1_tpxo9_atlas_30_v4','v_k1_tpxo9_atlas_30_v4',
                'v_n2_tpxo9_atlas_30_v4','v_m2_tpxo9_atlas_30_v4',
                'v_s2_tpxo9_atlas_30_v4','v_k2_tpxo9_atlas_30_v4',
                'v_m4_tpxo9_atlas_30_v4','v_ms4_tpxo9_atlas_30_v4',
                'v_mn4_tpxo9_atlas_30_v4','v_2n2_tpxo9_atlas_30_v4']
            self.model_file = {}
            for key,val in model_files.items():
                self.model_file[key] = [os.path.join(model_directory,
                    '{0}{1}{2}'.format(f,suffix,gzip)) for f in val]
            self.projection = '4326'
            self.scale = 1.0/100.0
            # model description and references
            self.reference = 'https://www.tpxo.net/global/tpxo9-atlas'
        elif (m == 'TPXO9.1'):
            self.grid_file = os.path.join(self.directory,
                'TPXO9.1','DATA','grid_tpxo9')
            self.model_file = dict(u=os.path.join(self.directory,
                'TPXO9.1','DATA','u_tpxo9.v1'))
            self.format = 'OTIS'
            self.projection = '4326'
            # model description and references
            self.reference = ('http://volkov.oce.orst.edu/tides/'
                'global.html')
        elif (m == 'TPXO8-atlas'):
            self.grid_file = os.path.join(self.directory,
                'tpxo8_atlas','grid_tpxo8atlas_30_v1')
            self.model_file = dict(u=os.path.join(self.directory,
                'tpxo8_atlas','uv.tpxo8_atlas_30_v1'))
            self.format = 'ATLAS'
            self.projection = '4326'
            # model description and references
            self.reference = ('http://volkov.oce.orst.edu/tides/'
                'tpxo8_atlas.html')
        elif (m == 'TPXO7.2'):
            self.grid_file = os.path.join(self.directory,
                'TPXO7.2_tmd','grid_tpxo7.2')
            self.model_file = dict(u=os.path.join(self.directory,
                'TPXO7.2_tmd','u_tpxo7.2'))
            self.format = 'OTIS'
            self.projection = '4326'
            # model description and references
            self.reference = ('http://volkov.oce.orst.edu/tides/'
                'global.html')
        elif (m == 'AODTM-5'):
            self.grid_file = os.path.join(self.directory,
                'aodtm5_tmd','grid_Arc5km')
            self.model_file = dict(u=os.path.join(self.directory,
                'aodtm5_tmd','UV0_Arc5km'))
            self.format = 'OTIS'
            self.projection = 'PSNorth'
            # model description and references
            self.reference = ('https://www.esr.org/research/'
                'polar-tide-models/list-of-polar-tide-models/'
                'aodtm-5/')
        elif (m == 'AOTIM-5'):
            self.grid_file = os.path.join(self.directory,
                'aotim5_tmd','grid_Arc5km')
            self.model_file = dict(u=os.path.join(self.directory,
                'aotim5_tmd','UV_Arc5km'))
            self.format = 'OTIS'
            self.projection = 'PSNorth'
            # model description and references
            self.reference = ('https://www.esr.org/research/'
                'polar-tide-models/list-of-polar-tide-models/'
                'aotim-5/')
        elif (m == 'AOTIM-5-2018'):
            self.grid_file = os.path.join(self.directory,
                'Arc5km2018','grid_Arc5km2018')
            self.model_file = dict(u=os.path.join(self.directory,
                'Arc5km2018','UV_Arc5km2018'))
            self.format = 'OTIS'
            self.projection = 'PSNorth'
            # model description and references
            self.reference = ('https://www.esr.org/research/'
                'polar-tide-models/list-of-polar-tide-models/'
                'aotim-5/')
        elif (m == 'Gr1km-v2'):
            self.grid_file = os.path.join(self.directory,
                'greenlandTMD_v2','grid_Greenland8.v2')
            self.model_file = dict(u=os.path.join(self.directory,
                'greenlandTMD_v2','u_Greenland8_rot.v2'))
            self.format = 'OTIS'
            self.projection = '3413'
            # model description and references
            self.reference = 'https://doi.org/10.1002/2016RG000546'
        elif (m == 'FES2014'):
            model_directory = {}
            model_directory['u'] = os.path.join(self.directory,
                'fes2014','eastward_velocity')
            model_directory['v'] = os.path.join(self.directory,
                'fes2014','northward_velocity')
            model_files = ['2n2.nc','eps2.nc','j1.nc','k1.nc',
                'k2.nc','l2.nc','la2.nc','m2.nc','m3.nc','m4.nc',
                'm6.nc','m8.nc','mf.nc','mks2.nc','mm.nc',
                'mn4.nc','ms4.nc','msf.nc','msqm.nc','mtm.nc',
                'mu2.nc','n2.nc','n4.nc','nu2.nc','o1.nc','p1.nc',
                'q1.nc','r2.nc','s1.nc','s2.nc','s4.nc','sa.nc',
                'ssa.nc','t2.nc']
            self.model_file = {}
            for key,val in model_directory.items():
                self.model_file[key] = [os.path.join(model_directory,
                    '{0}{1}'.format(f,gzip)) for f in val]
            self.constituents = ['2n2','eps2','j1','k1','k2','l2','lambda2',
                'm2','m3','m4','m6','m8','mf','mks2','mm','mn4','ms4','msf',
                'msqm','mtm','mu2','n2','n4','nu2','o1','p1','q1','r2','s1',
                's2','s4','sa','ssa','t2']
            self.format = 'FES'
            self.scale = 1.0
            # model description and references
            self.reference = ('https://www.aviso.altimetry.fr/en/data/products'
                'auxiliary-products/global-tide-fes.html')
        else:
            raise Exception("Unlisted tide model")
        # return the model parameters
        return self

    def from_file(self, parameter_file):
        """Create a model object from an input definition file
        """
        # variable with parameter definitions
        parameters = {}
        # Opening parameter file and assigning file ID number (fid)
        fid = open(parameter_file, 'r')
        # for each line in the file will extract the parameter (name and value)
        for fileline in fid:
            # Splitting the input line between parameter name and value
            part = fileline.rstrip().split(maxsplit=1)
            # filling the parameter definition variable
            parameters[part[0]] = part[1]
        # close the parameter file
        fid.close()
        # convert from dictionary to model variable
        temp = self.from_dict(parameters)
        # verify model type
        assert temp.type in ('OTIS','ATLAS','netcdf','GOT','FES')
        # convert scale from string to float
        temp.scale = float(temp.scale)
        # split model file into list if an ATLAS, GOT or FES file
        # model files can be comma, tab or space delimited
        if re.search(r'[\s\,]+', temp.model_file):
            temp.model_file = re.split(r'[\s\,]+',temp.model_file)
        # make sure necessary keys are with model type
        if temp.type in ('OTIS','ATLAS'):
            assert temp.projection
        return temp

    def from_dict(self,d):
        """Create a model object from a python dictionary
        """
        for key,val in d.items():
            setattr(self,key,copy.copy(val))
        # return the model parameters
        return self
