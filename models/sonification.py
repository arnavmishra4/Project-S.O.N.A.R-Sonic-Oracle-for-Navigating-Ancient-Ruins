#Cell 1: Setup & Global Configuration
import rasterio
import numpy as np
import soundfile as sf # For efficient reading/writing of WAV data
from pydub import AudioSegment # Primarily for per-cell segment creation and panning
from pydub.effects import normalize, compress_dynamic_range # Will be used carefully or noted
import os
import glob
import shutil # For cleaning up temporary directories
from scipy.ndimage import uniform_filter
from scipy.signal import butter, lfilter
import matplotlib.pyplot as plt
from shapely.geometry import box, mapping, Polygon # For geometry operations
from rasterio.mask import mask # For clipping rasters
from pyproj import CRS, Transformer # For coordinate transformations
import json # For saving geospatial metadata
import rasterio.merge # For mosaicking DTM tiles
import re # For regex to parse DTM file prefixes
from rasterio.transform import array_bounds # Import for calculating bounds from profile
import os
from config import (
    LIDAR_DTM_TILES_DIR,
    HYDRO_GLOBAL_BASE_DIR,
    HYDRO_GLOBAL_FILES,
    SONIFIED_AUDIO_BASE_DIR,
    AUDIO_SAMPLE_RATE,
    DURATION_PER_GRID_CELL,
)

# New: Class to store cell geometry and audio timing (Moved to global scope)
class CellGeom:
    def __init__(self, minx, miny, maxx, maxy, audio_start_ms, audio_end_ms):
        self.minx = minx
        self.miny = miny
        self.maxx = maxx
        self.maxy = maxy
        self.audio_start_ms = audio_start_ms
        self.audio_end_ms = audio_end_ms

    def to_dict(self):
        return {
            "minx": self.minx,
            "miny": self.miny,
            "maxx": self.maxx,
            "maxy": self.maxy,
            "audio_start_ms": self.audio_start_ms,
            "audio_end_ms": self.audio_end_ms
        }

LIDAR_DTM_TILES_DIR = LIDAR_DTM_TILES_DIR
HYDRO_GLOBAL_BASE_DIR = HYDRO_GLOBAL_BASE_DIR # Assuming global HydroSHEDS files are here
HYDRO_GLOBAL_FILES = {
    "conditioned_dem": "/kaggle/input/south-america-hydroshed-dataset/sa_con_3s/sa_con_3s.tif",
    "flow_direction": "/kaggle/input/south-america-hydroshed-dataset/sa_dir_3s/sa_dir_3s.tif",
    "flow_accumulation": "/kaggle/input/south-america-hydroshed-dataset/sa_acc_3s/sa_acc_3s.tif"
}

LIDAR_REFERENCE_CODES = {
    'BR_AC_10': ['HUM'], 'BR_RO_05': ['RIB'], 'BR_PA_02': ['TAL'],
    'BR_AC_07': ['ANT', 'BON'], 'BR_AC_09': ['BON', 'HUM'], 'BR_AM_04': ['DUC'],
    'BR_PA_04': ['BA3'], 'BR_RO_03': ['BA3', 'B38'], 'BR_MT_01': ['BA3'],
    'BR_AM_03': ['DUC'],
}

AOI_CENTERS_DEGREES = {
    'BR_AC_10': {'lat': -10.0, 'lon': -68.0, 'buffer_km': 25}, 'BR_RO_05': {'lat': -10.5, 'lon': -63.5, 'buffer_km': 25},
    'BR_PA_02': {'lat': -6.0, 'lon': -52.0, 'buffer_km': 25}, 'BR_AC_07': {'lat': -9.9, 'lon': -67.8, 'buffer_km': 25},
    'BR_AC_09': {'lat': -10.1, 'lon': -67.9, 'buffer_km': 25}, 'BR_AM_04': {'lat': -3.5, 'lon': -59.5, 'buffer_km': 25},
    'BR_PA_04': {'lat': -3.0, 'lon': -53.0, 'buffer_km': 25}, 'BR_RO_03': {'lat': -10.0, 'lon': -63.0, 'buffer_km': 25},
    'BR_MT_01': {'lat': -12.0, 'lon': -60.0, 'buffer_km': 25}, 'BR_AM_03': {'lat': -3.0, 'lon': -60.0, 'buffer_km': 25},
}

# --- Output Directories ---
output_audio_base_dir = "sonified_outputs" # Master output for all sonified WAVs
output_viz_base_dir = "geospatial_visualizations" # Master output for all visualization images
output_clipped_hydro_dir = "clipped_hydrosheds_50km" # Output for clipped HydroSHEDS files
output_gee_exports_dir = "/kaggle/input/sonic-geetiffs-50km" # EXPECT GEE exports to be downloaded here in Kaggle datasets

os.makedirs(output_audio_base_dir, exist_ok=True)
os.makedirs(output_viz_base_dir, exist_ok=True)
os.makedirs(output_clipped_hydro_dir, exist_ok=True)


# --- Site Categorization for Sonification & Naming ---
# THESE GLOBAL LISTS MUST BE DEFINED AT THE TOP LEVEL OF THE CELL
ARCHAEOLOGICAL_TRANSECTS = ['BR_AC_10', 'BR_RO_05', 'BR_PA_02', 'BR_AC_07', 'BR_AC_09']
JUNGLE_TRANSECTS = ['BR_AM_04', 'BR_PA_04', 'BR_RO_03', 'BR_MT_01']
CITY_TRANSECTS = ['BR_AM_03']

TRANSECTS_TO_PROCESS = ARCHAEOLOGICAL_TRANSECTS + JUNGLE_TRANSECTS + CITY_TRANSECTS


DTM_SERIES_MAP = {
    'BR_AC_10': 'HUM_A01_2013_laz_', 'BR_RO_05': 'RIB_A01_2014_laz_', 'BR_PA_02': 'TAL_A01_2013_laz_',
    'BR_AC_07': 'HUM_A01_2013_laz_', 'BR_AC_09': 'BON_A01_2013_laz_', 'BR_AM_04': 'TAP_A01_2012_laz_',
    'BR_PA_04': 'BA3_A01_2014_laz_', 'BR_RO_03': 'BA3_A01_2014_laz_', 'BR_MT_01': 'BA3_A02_2014_laz_',
    'BR_AM_03': 'DUC_A01_2012_laz_',
}

# --- Sonification & Musical Parameters ---
TARGET_AUDIO_DURATION = 7.0
SAMPLE_RATE = 11025 # Aggressively reduced sample rate for max memory efficiency on Kaggle
DURATION_PER_GRID_CELL = 6.0
PROCESSING_GRID_SIZE_METERS = 50

GLOBAL_NDVI_PITCH_HIGH_MIN_MIDI = 70
GLOBAL_NDVI_PITCH_HIGH_MAX_MIDI = 85
GLOBAL_NDVI_PITCH_LOW_MIN_MIDI = 30
GLOBAL_NDVI_PITCH_LOW_MAX_MIDI = 45

GLOBAL_ROUGHNESS_FILTER_HIGH_MIN = 500
GLOBAL_ROUGHNESS_FILTER_HIGH_MAX = 15000
GLOBAL_ROUGHNESS_FILTER_LOW_MIN = 30
GLOBAL_ROUGHNESS_FILTER_LOW_MAX = 2000

GLOBAL_HYDRO_DRONE_HIGH_MIN = 100
GLOBAL_HYDRO_DRONE_HIGH_MAX = 300
GLOBAL_HYDRO_DRONE_LOW_MIN = 30
GLOBAL_HYDRO_DRONE_LOW_MAX = 100


MAJOR_SCALE_MIDI = [60, 62, 64, 65, 67, 69, 71, 72]
MINOR_PENTATONIC_MIDI = [60, 63, 65, 67, 70, 72]
CHROMATIC_SCALE_MIDI = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

ANOMALY_GLISS_MIDI_START = 72
ANOMALY_GLISS_MIDI_END = 84
ANOMALY_PING_MIDI_NOTE = 96
ANOMALY_PING_DURATION = 0.5
ANOMALY_PING_AMPLITUDE = 1.0

NDWI_WATER_THRESHOLD = 0.2
WATER_BODY_SUPPRESSION_GAIN_DB = -15.0
HYDRO_BOOST_GAIN_DB = 5.0

# --- NEW: Global cache for CRS Transformers ---
crs_transformers_cache = {}

print("Cell 1: Setup & Global Configuration Complete.")


# -----------------------------------------------------------------------------
# Configuration & File Paths (YOU MUST UPDATE THESE PATHS TO YOUR ACTUAL FILES)
# This dictionary now specifies the FULL LIST OF DTM TILES for each series
# and your downloaded/uploaded satellite and hydrological data paths.
# -----------------------------------------------------------------------------
TRANSECT_FILE_PATHS = {
    'BR_AC_10': { # Uses HUM_A01_2013_laz_ series
        'dtm_tile_paths_list': [ # Explicitly list all DTM tiles for this series
            r"/kaggle/input/nasa-amazon-lidar-2008-2018/Nasa_lidar_2008_to_2018_DTMs/DTM_tiles/HUM_A01_2013_laz_0.tif",
            r"/kaggle/input/nasa-amazon-lidar-2008-2018/Nasa_lidar_2008_to_2018_DTMs/DTM_tiles/HUM_A01_2013_laz_1.tif",
            r"/kaggle/input/nasa-amazon-lidar-2008-2018/Nasa_lidar_2008_to_2018_DTMs/DTM_tiles/HUM_A01_2013_laz_10.tif",
            r"/kaggle/input/nasa-amazon-lidar-2008-2018/Nasa_lidar_2008_to_2018_DTMs/DTM_tiles/HUM_A01_2013_laz_11.tif",
            r"/kaggle/input/nasa-amazon-lidar-2008-2018/Nasa_lidar_2008_to_2018_DTMs/DTM_tiles/HUM_A01_2013_laz_2.tif",
            r"/kaggle/input/nasa-amazon-lidar-2008-2018/Nasa_lidar_2008_to_2018_DTMs/DTM_tiles/HUM_A01_2013_laz_3.tif",
            r"/kaggle/input/nasa-amazon-lidar-2008-2018/Nasa_lidar_2008_to_2018_DTMs/DTM_tiles/HUM_A01_2013_laz_4.tif",
            r"/kaggle/input/nasa-amazon-lidar-2008-2018/Nasa_lidar_2008_to_2018_DTMs/DTM_tiles/HUM_A01_2013_laz_5.tif",
            r"/kaggle/input/nasa-amazon-lidar-2008-2018/Nasa_lidar_2008_to_2018_DTMs/DTM_tiles/HUM_A01_2013_laz_6.tif",
            r"/kaggle/input/nasa-amazon-lidar-2008-2018/Nasa_lidar_2008_to_2018_DTMs/DTM_tiles/HUM_A01_2013_laz_7.tif",
            r"/kaggle/input/nasa-amazon-lidar-2008-2018/Nasa_lidar_2008_to_2018_DTMs/DTM_tiles/HUM_A01_2013_laz_8.tif",
            r"/kaggle/input/nasa-amazon-lidar-2008-2018/Nasa_lidar_2008_to_2018_DTMs/DTM_tiles/HUM_A01_2013_laz_9.tif",
        ],
        'sat_30m_dry': r"/kaggle/input/weeewooo-new-satellite-dataset/KAS_HUM_A01_2013_S2_DrySeason_50km.tif",
        'sat_30m_wet': r"/kaggle/input/weeewooo-new-satellite-dataset/KAS_HUM_A01_2013_S2_WetSeason_50km.tif",
        'hydro_dem': r"/kaggle/input/ultra-final-hydrological-dataset/hydro_extracts/HUM_A01_2013_conditioning_clipped.tif",
        'hydro_flow_dir': r"/kaggle/input/ultra-final-hydrological-dataset/hydro_extracts/HUM_A01_2013_flow_direction_clipped.tif",
        'hydro_flow_acc': r"/kaggle/input/ultra-final-hydrological-dataset/hydro_extracts/HUM_A01_2013_flow_accumulation_clipped.tif"
    },
    'BR_RO_05': { # Uses RIB_A01_2014_laz_ series
        'dtm_tile_paths_list': [ # Explicitly list all DTM tiles for this series
            r"/kaggle/input/nasa-amazon-lidar-2008-2018/Nasa_lidar_2008_to_2018_DTMs/DTM_tiles/RIB_A01_2014_laz_0.tif",
            r"/kaggle/input/nasa-amazon-lidar-2008-2018/Nasa_lidar_2008_to_2018_DTMs/DTM_tiles/RIB_A01_2014_laz_1.tif",
            r"/kaggle/input/nasa-amazon-lidar-2008-2018/Nasa_lidar_2008_to_2018_DTMs/DTM_tiles/RIB_A01_2014_laz_10.tif",
            r"/kaggle/input/nasa-amazon-lidar-2008-2018/Nasa_lidar_2008_to_2018_DTMs/DTM_tiles/RIB_A01_2014_laz_11.tif",
            r"/kaggle/input/nasa-amazon-lidar-2008-2018/Nasa_lidar_2008_to_2018_DTMs/DTM_tiles/RIB_A01_2014_laz_12.tif",
            r"/kaggle/input/nasa-amazon-lidar-2008-2018/Nasa_lidar_2008_to_2018_DTMs/DTM_tiles/RIB_A01_2014_laz_13.tif",
            r"/kaggle/input/nasa-amazon-lidar-2008-2018/Nasa_lidar_2008_to_2018_DTMs/DTM_tiles/RIB_A01_2014_laz_14.tif",
            r"/kaggle/input/nasa-amazon-lidar-2008-2018/Nasa_lidar_2008_to_2018_DTMs/DTM_tiles/RIB_A01_2014_laz_15.tif",
            r"/kaggle/input/nasa-amazon-lidar-2008-2018/Nasa_lidar_2008_to_2018_DTMs/DTM_tiles/RIB_A01_2014_laz_16.tif",
            r"/kaggle/input/nasa-amazon-lidar-2008-2018/Nasa_lidar_2008_to_2018_DTMs/DTM_tiles/RIB_A01_2014_laz_17.tif",
            r"/kaggle/input/nasa-amazon-lidar-2008-2018/Nasa_lidar_2008_to_2018_DTMs/DTM_tiles/RIB_A01_2014_laz_2.tif",
            r"/kaggle/input/nasa-amazon-lidar-2008-2018/Nasa_lidar_2008_to_2018_DTMs/DTM_tiles/RIB_A01_2014_laz_3.tif",
            r"/kaggle/input/nasa-amazon-lidar-2008-2018/Nasa_lidar_2008_to_2018_DTMs/DTM_tiles/RIB_A01_2014_laz_4.tif",
            r"/kaggle/input/nasa-amazon-lidar-2008-2018/Nasa_lidar_2008_to_2018_DTMs/DTM_tiles/RIB_A01_2014_laz_5.tif",
            r"/kaggle/input/nasa-amazon-lidar-2008-2018/Nasa_lidar_2008_to_2018_DTMs/DTM_tiles/RIB_A01_2014_laz_6.tif",
            r"/kaggle/input/nasa-amazon-lidar-2008-2018/Nasa_lidar_2008_to_2018_DTMs/DTM_tiles/RIB_A01_2014_laz_8.tif",
            r"/kaggle/input/nasa-amazon-lidar-2008-2018/Nasa_lidar_2008_to_2018_DTMs/DTM_tiles/RIB_A01_2014_laz_9.tif",
        ],
        'sat_30m_dry': r"/kaggle/input/weeewooo-new-satellite-dataset/KAS_RIB_A01_2014_S2_DrySeason_50km.tif",
        'sat_30m_wet': r"/kaggle/input/weeewooo-new-satellite-dataset/KAS_RIB_A01_2014_S2_WetSeason_50km.tif",
        'hydro_dem': r"/kaggle/input/ultra-final-hydrological-dataset/hydro_extracts/RIB_A01_2014_conditioning_clipped.tif",
        'hydro_flow_dir': r"/kaggle/input/ultra-final-hydrological-dataset/hydro_extracts/RIB_A01_2014_flow_direction_clipped.tif",
        'hydro_flow_acc': r"/kaggle/input/ultra-final-hydrological-dataset/hydro_extracts/RIB_A01_2014_flow_accumulation_clipped.tif"
    },
    'BR_PA_02': { # Uses TAL_A01_2013_laz_ series
        'dtm_tile_paths_list': [ # Explicitly list all DTM tiles for this series
            r"/kaggle/input/nasa-amazon-lidar-2008-2018/Nasa_lidar_2008_to_2018_DTMs/DTM_tiles/TAL_A01_2013_laz_0.tif",
            r"/kaggle/input/nasa-amazon-lidar-2008-2018/Nasa_lidar_2008_to_2018_DTMs/DTM_tiles/TAL_A01_2013_laz_1.tif",
            r"/kaggle/input/nasa-amazon-lidar-2008-2018/Nasa_lidar_2008_to_2018_DTMs/DTM_tiles/TAL_A01_2013_laz_10.tif",
            r"/kaggle/input/nasa-amazon-lidar-2008-2018/Nasa_lidar_2008_to_2018_DTMs/DTM_tiles/TAL_A01_2013_laz_11.tif",
            r"/kaggle/input/nasa-amazon-lidar-2008-2018/Nasa_lidar_2008_to_2018_DTMs/DTM_tiles/TAL_A01_2013_laz_2.tif",
            r"/kaggle/input/nasa-amazon-lidar-2008-2018/Nasa_lidar_2008_to_2018_DTMs/DTM_tiles/TAL_A01_2013_laz_3.tif",
            r"/kaggle/input/nasa-amazon-lidar-2008-2018/Nasa_lidar_2008_to_2018_DTMs/DTM_tiles/TAL_A01_2013_laz_4.tif",
            r"/kaggle/input/nasa-amazon-lidar-2008-2018/Nasa_lidar_2008_to_2018_DTMs/DTM_tiles/TAL_A01_2013_laz_5.tif",
            r"/kaggle/input/nasa-amazon-lidar-2008-2018/Nasa_lidar_2008_to_2018_DTMs/DTM_tiles/TAL_A01_2013_laz_6.tif",
            r"/kaggle/input/nasa-amazon-lidar-2008-2018/Nasa_lidar_2008_to_2018_DTMs/DTM_tiles/TAL_A01_2013_laz_7.tif",
            r"/kaggle/input/nasa-amazon-lidar-2008-2018/Nasa_lidar_2008_to_2018_DTMs/DTM_tiles/TAL_A01_2013_laz_8.tif",
            r"/kaggle/input/nasa-amazon-lidar-2008-2018/Nasa_lidar_2008_to_2018_DTMs/DTM_tiles/TAL_A01_2013_laz_9.tif",
        ],
        'sat_30m_dry': r"/kaggle/input/weeewooo-new-satellite-dataset/KAS_TAL_A01_2013_S2_DrySeason_50km.tif",
        'sat_30m_wet': r'/kaggle/input/weeewooo-new-satellite-dataset/KAS_TAL_A01_2013_S2_WetSeason_50km.tif',
        'hydro_dem': r"/kaggle/input/ultra-final-hydrological-dataset/hydro_extracts/TAL_A01_2013_conditioning_clipped.tif",
        'hydro_flow_dir': r"/kaggle/input/ultra-final-hydrological-dataset/hydro_extracts/TAL_A01_2013_flow_direction_clipped.tif",
        'hydro_flow_acc': r"/kaggle/input/ultra-final-hydrological-dataset/hydro_extracts/TAL_A01_2013_flow_accumulation_clipped.tif"
    },
    'BR_AC_07': { # Uses HUM_A01_2013_laz_ series (same as BR_AC_10)
        'dtm_tile_paths_list': [ # Explicitly list all DTM tiles for this series
            r"/kaggle/input/nasa-amazon-lidar-2008-2018/Nasa_lidar_2008_to_2018_DTMs/DTM_tiles/HUM_A01_2013_laz_0.tif",
            r"/kaggle/input/nasa-amazon-lidar-2008-2018/Nasa_lidar_2008_to_2018_DTMs/DTM_tiles/HUM_A01_2013_laz_1.tif",
            r"/kaggle/input/nasa-amazon-lidar-2008-2018/Nasa_lidar_2008_to_2018_DTMs/DTM_tiles/HUM_A01_2013_laz_10.tif",
            r"/kaggle/input/nasa-amazon-lidar-2008-2018/Nasa_lidar_2008_to_2018_DTMs/DTM_tiles/HUM_A01_2013_laz_11.tif",
            r"/kaggle/input/nasa-amazon-lidar-2008-2018/Nasa_lidar_2008_to_2018_DTMs/DTM_tiles/HUM_A01_2013_laz_2.tif",
            r"/kaggle/input/nasa-amazon-lidar-2008-2018/Nasa_lidar_2008_to_2018_DTMs/DTM_tiles/HUM_A01_2013_laz_3.tif",
            r"/kaggle/input/nasa-amazon-lidar-2008-2018/Nasa_lidar_2008_to_2018_DTMs/DTM_tiles/HUM_A01_2013_laz_4.tif",
            r"/kaggle/input/nasa-amazon-lidar-2008-2018/Nasa_lidar_2008_to_2018_DTMs/DTM_tiles/HUM_A01_2013_laz_5.tif",
            r"/kaggle/input/nasa-amazon-lidar-2008-2018/Nasa_lidar_2008_to_2018_DTMs/DTM_tiles/HUM_A01_2013_laz_6.tif",
            r"/kaggle/input/nasa-amazon-lidar-2008-2018/Nasa_lidar_2008_to_2018_DTMs/DTM_tiles/HUM_A01_2013_laz_7.tif",
            r"/kaggle/input/nasa-amazon-lidar-2008-2018/Nasa_lidar_2008_to_2018_DTMs/DTM_tiles/HUM_A01_2013_laz_8.tif",
            r"/kaggle/input/nasa-amazon-lidar-2008-2018/Nasa_lidar_2008_to_2018_DTMs/DTM_tiles/HUM_A01_2013_laz_9.tif",
        ],
        'sat_30m_dry': r"/kaggle/input/weeewooo-new-satellite-dataset/KAS_HUM_A01_2013_S2_DrySeason_50km.tif",
        'sat_30m_wet': r"/kaggle/input/weeewooo-new-satellite-dataset/KAS_HUM_A01_2013_S2_WetSeason_50km.tif",
        'hydro_dem': r"/kaggle/input/ultra-final-hydrological-dataset/hydro_extracts/HUM_A01_2013_conditioning_clipped.tif",
        'hydro_flow_dir': r"/kaggle/input/ultra-final-hydrological-dataset/hydro_extracts/HUM_A01_2013_flow_direction_clipped.tif",
        'hydro_flow_acc': r"/kaggle/input/ultra-final-hydrological-dataset/hydro_extracts/HUM_A01_2013_flow_accumulation_clipped.tif"
    },
    'BR_AC_09': { # Uses BON_A01_2013_laz_ series
        'dtm_tile_paths_list': [ # Explicitly list all DTM tiles for this series
            r"/kaggle/input/nasa-amazon-lidar-2008-2018/Nasa_lidar_2008_to_2018_DTMs/DTM_tiles/BON_A01_2013_laz_0.tif",
            r"/kaggle/input/nasa-amazon-lidar-2008-2018/Nasa_lidar_2008_to_2018_DTMs/DTM_tiles/BON_A01_2013_laz_1.tif",
            r"/kaggle/input/nasa-amazon-lidar-2008-2018/Nasa_lidar_2008_to_2018_DTMs/DTM_tiles/BON_A01_2013_laz_10.tif",
            r"/kaggle/input/nasa-amazon-lidar-2008-2018/Nasa_lidar_2008_to_2018_DTMs/DTM_tiles/BON_A01_2013_laz_11.tif",
            r"/kaggle/input/nasa-amazon-lidar-2008-2018/Nasa_lidar_2008_to_2018_DTMs/DTM_tiles/BON_A01_2013_laz_12.tif",
            r"/kaggle/input/nasa-amazon-lidar-2008-2018/Nasa_lidar_2008_to_2018_DTMs/DTM_tiles/BON_A01_2013_laz_13.tif",
            r"/kaggle/input/nasa-amazon-lidar-2008-2018/Nasa_lidar_2008_to_2018_DTMs/DTM_tiles/BON_A01_2013_laz_2.tif",
            r"/kaggle/input/nasa-amazon-lidar-2008-2018/Nasa_lidar_2008_to_2018_DTMs/DTM_tiles/BON_A01_2013_laz_3.tif",
            r"/kaggle/input/nasa-amazon-lidar-2008-2018/Nasa_lidar_2008_to_2018_DTMs/DTM_tiles/BON_A01_2013_laz_4.tif",
            r"/kaggle/input/nasa-amazon-lidar-2008-2018/Nasa_lidar_2008_to_2018_DTMs/DTM_tiles/BON_A01_2013_laz_5.tif",
            r"/kaggle/input/nasa-amazon-lidar-2008-2018/Nasa_lidar_2008_to_2018_DTMs/DTM_tiles/BON_A01_2013_laz_6.tif",
            r"/kaggle/input/nasa-amazon-lidar-2008-2018/Nasa_lidar_2008_to_2018_DTMs/DTM_tiles/BON_A01_2013_laz_7.tif",
            r"/kaggle/input/nasa-amazon-lidar-2008-2018/Nasa_lidar_2008_to_2018_DTMs/DTM_tiles/BON_A01_2013_laz_8.tif",
            r"/kaggle/input/nasa-amazon-lidar-2008-2018/Nasa_lidar_2008_to_2018_DTMs/DTM_tiles/BON_A01_2013_laz_9.tif",
        ],
        'sat_30m_dry': r"/kaggle/input/weeewooo-new-satellite-dataset/KAS_BON_A01_2013_S2_DrySeason_50km.tif",
        'sat_30m_wet': r"/kaggle/input/weeewooo-new-satellite-dataset/KAS_BON_A01_2013_S2_WetSeason_50km.tif",
        'hydro_dem': r"/kaggle/input/ultra-final-hydrological-dataset/hydro_extracts/BON_A01_2013_conditioning_clipped.tif",
        'hydro_flow_dir': r"/kaggle/input/ultra-final-hydrological-dataset/hydro_extracts/BON_A01_2013_flow_direction_clipped.tif",
        'hydro_flow_acc': r"/kaggle/input/ultra-final-hydrological-dataset/hydro_extracts/BON_A01_2013_flow_accumulation_clipped.tif"
    },
    'BR_AM_04': { # Uses TAP_A01_2012_laz_ series
        'dtm_tile_paths_list': [ # Explicitly list all DTM tiles for this series
            r"/kaggle/input/nasa-amazon-lidar-2008-2018/Nasa_lidar_2008_to_2018_DTMs/DTM_tiles/TAP_A01_2012_laz_0.tif",
            r"/kaggle/input/nasa-amazon-lidar-2008-2018/Nasa_lidar_2008_to_2018_DTMs/DTM_tiles/TAP_A01_2012_laz_1.tif",
            r"/kaggle/input/nasa-amazon-lidar-2008-2018/Nasa_lidar_2008_to_2018_DTMs/DTM_tiles/TAP_A01_2012_laz_10.tif",
            r"/kaggle/input/nasa-amazon-lidar-2008-2018/Nasa_lidar_2008_to_2018_DTMs/DTM_tiles/TAP_A01_2012_laz_11.tif",
            r"/kaggle/input/nasa-amazon-lidar-2008-2018/Nasa_lidar_2008_to_2018_DTMs/DTM_tiles/TAP_A01_2012_laz_12.tif",
            r"/kaggle/input/nasa-amazon-lidar-2008-2018/Nasa_lidar_2008_to_2018_DTMs/DTM_tiles/TAP_A01_2012_laz_13.tif",
            r"/kaggle/input/nasa-amazon-lidar-2008-2018/Nasa_lidar_2008_to_2018_DTMs/DTM_tiles/TAP_A01_2012_laz_14.tif",
            r"/kaggle/input/nasa-amazon-lidar-2008-2018/Nasa_lidar_2008_to_2018_DTMs/DTM_tiles/TAP_A01_2012_laz_15.tif",
            r"/kaggle/input/nasa-amazon-lidar-2008-2018/Nasa_lidar_2008_to_2018_DTMs/DTM_tiles/TAP_A01_2012_laz_16.tif",
            r"/kaggle/input/nasa-amazon-lidar-2008-2018/Nasa_lidar_2008_to_2018_DTMs/DTM_tiles/TAP_A01_2012_laz_2.tif",
            r"/kaggle/input/nasa-amazon-lidar-2008-2018/Nasa_lidar_2008_to_2018_DTMs/DTM_tiles/TAP_A01_2012_laz_3.tif",
            r"/kaggle/input/nasa-amazon-lidar-2008-2018/Nasa_lidar_2008_to_2018_DTMs/DTM_tiles/TAP_A01_2012_laz_4.tif",
            r"/kaggle/input/nasa-amazon-lidar-2008-2018/Nasa_lidar_2008_to_2018_DTMs/DTM_tiles/TAP_A01_2012_laz_5.tif",
            r"/kaggle/input/nasa-amazon-lidar-2008-2018/Nasa_lidar_2008_to_2018_DTMs/DTM_tiles/TAP_A01_2012_laz_6.tif",
            r"/kaggle/input/nasa-amazon-lidar-2008-2018/Nasa_lidar_2008_to_2018_DTMs/DTM_tiles/TAP_A01_2012_laz_7.tif",
            r"/kaggle/input/nasa-amazon-lidar-2008-2018/Nasa_lidar_2008_to_2018_DTMs/DTM_tiles/TAP_A01_2012_laz_8.tif",
            r"/kaggle/input/nasa-amazon-lidar-2008-2018/Nasa_lidar_2008_to_2018_DTMs/DTM_tiles/TAP_A01_2012_laz_9.tif",
        ],
        'sat_30m_dry': r"/kaggle/input/weeewooo-new-satellite-dataset/KAS_TAP_A01_2012_S2_DrySeason_50km.tif",
        'sat_30m_wet': r"/kaggle/input/weeewooo-new-satellite-dataset/KAS_TAP_A01_2012_S2_WetSeason_50km.tif",
        'hydro_dem': r"/kaggle/input/ultra-final-hydrological-dataset/hydro_extracts/TAP_A01_2012_conditioning_clipped.tif",
        'hydro_flow_dir': r"/kaggle/input/ultra-final-hydrological-dataset/hydro_extracts/TAP_A01_2012_flow_direction_clipped.tif",
        'hydro_flow_acc': r"/kaggle/input/ultra-final-hydrological-dataset/hydro_extracts/TAP_A01_2012_flow_accumulation_clipped.tif"
    },
    'BR_PA_04': { # Uses BA3_A01_2014_laz_ series
        'dtm_tile_paths_list': [ # Explicitly list all DTM tiles for this series
            r"/kaggle/input/nasa-amazon-lidar-2008-2018/Nasa_lidar_2008_to_2018_DTMs/DTM_tiles/BA3_A01_2014_laz_0.tif",
            r"/kaggle/input/nasa-amazon-lidar-2008-2018/Nasa_lidar_2008_to_2018_DTMs/DTM_tiles/BA3_A01_2014_laz_1.tif",
            r"/kaggle/input/nasa-amazon-lidar-2008-2018/Nasa_lidar_2008_to_2018_DTMs/DTM_tiles/BA3_A01_2014_laz_10.tif",
            r"/kaggle/input/nasa-amazon-lidar-2008-2018/Nasa_lidar_2008_to_2018_DTMs/DTM_tiles/BA3_A01_2014_laz_11.tif",
            r"/kaggle/input/nasa-amazon-lidar-2008-2018/Nasa_lidar_2008_to_2018_DTMs/DTM_tiles/BA3_A01_2014_laz_2.tif",
            r"/kaggle/input/nasa-amazon-lidar-2008-2018/Nasa_lidar_2008_to_2018_DTMs/DTM_tiles/BA3_A01_2014_laz_3.tif",
            r"/kaggle/input/nasa-amazon-lidar-2008-2018/Nasa_lidar_2008_to_2018_DTMs/DTM_tiles/BA3_A01_2014_laz_4.tif",
            r"/kaggle/input/nasa-amazon-lidar-2008-2018/Nasa_lidar_2008_to_2018_DTMs/DTM_tiles/BA3_A01_2014_laz_5.tif",
            r"/kaggle/input/nasa-amazon-lidar-2008-2018/Nasa_lidar_2008_to_2018_DTMs/DTM_tiles/BA3_A01_2014_laz_6.tif",
            r"/kaggle/input/nasa-amazon-lidar-2008-2018/Nasa_lidar_2008_to_2018_DTMs/DTM_tiles/BA3_A01_2014_laz_7.tif",
            r"/kaggle/input/nasa-amazon-lidar-2008-2018/Nasa_lidar_2008_to_2018_DTMs/DTM_tiles/BA3_A01_2014_laz_8.tif",
            r"/kaggle/input/nasa-amazon-lidar-2008-2018/Nasa_lidar_2008_to_2018_DTMs/DTM_tiles/BA3_A01_2014_laz_9.tif",
        ],
        'sat_30m_dry': r"/kaggle/input/weeewooo-new-satellite-dataset/KAS_BA3_A01_2014_S2_DrySeason_50km.tif",
        'sat_30m_wet': r'/kaggle/input/weeewooo-new-satellite-dataset/KAS_BA3_A01_2014_S2_WetSeason_50km.tif',
        'hydro_dem': r"/kaggle/input/ultra-final-hydrological-dataset/hydro_extracts/BA3_A01_2014_conditioning_clipped.tif",
        'hydro_flow_dir': r"/kaggle/input/ultra-final-hydrological-dataset/hydro_extracts/BA3_A01_2014_flow_direction_clipped.tif",
        'hydro_flow_acc': r"/kaggle/input/ultra-final-hydrological-dataset/hydro_extracts/BA3_A01_2014_flow_accumulation_clipped.tif"
    },
    'BR_RO_03': { # Uses BA3_A01_2014_laz_ series (same as BR_PA_04)
        'dtm_tile_paths_list': [ # Explicitly list all DTM tiles for this series
            r"/kaggle/input/nasa-amazon-lidar-2008-2018/Nasa_lidar_2008_to_2018_DTMs/DTM_tiles/BA3_A01_2014_laz_0.tif",
            r"/kaggle/input/nasa-amazon-lidar-2008-2018/Nasa_lidar_2008_to_2018_DTMs/DTM_tiles/BA3_A01_2014_laz_1.tif",
            r"/kaggle/input/nasa-amazon-lidar-2008-2018/Nasa_lidar_2008_to_2018_DTMs/DTM_tiles/BA3_A01_2014_laz_10.tif",
            r"/kaggle/input/nasa-amazon-lidar-2008-2018/Nasa_lidar_2008_to_2018_DTMs/DTM_tiles/BA3_A01_2014_laz_11.tif",
            r"/kaggle/input/nasa-amazon-lidar-2008-2018/Nasa_lidar_2008_to_2018_DTMs/DTM_tiles/BA3_A01_2014_laz_2.tif",
            r"/kaggle/input/nasa-amazon-lidar-2008-2018/Nasa_lidar_2008_to_2018_DTMs/DTM_tiles/BA3_A01_2014_laz_3.tif",
            r"/kaggle/input/nasa-amazon-lidar-2008-2018/Nasa_lidar_2008_to_2018_DTMs/DTM_tiles/BA3_A01_2014_laz_4.tif",
            r"/kaggle/input/nasa-amazon-lidar-2008-2018/Nasa_lidar_2008_to_2018_DTMs/DTM_tiles/BA3_A01_2014_laz_5.tif",
            r"/kaggle/input/nasa-amazon-lidar-2008-2018/Nasa_lidar_2008_to_2018_DTMs/DTM_tiles/BA3_A01_2014_laz_6.tif",
            r"/kaggle/input/nasa-amazon-lidar-2008-2018/Nasa_lidar_2008_to_2018_DTMs/DTM_tiles/BA3_A01_2014_laz_7.tif",
            r"/kaggle/input/nasa-amazon-lidar-2008-2018/Nasa_lidar_2008_to_2018_DTMs/DTM_tiles/BA3_A01_2014_laz_8.tif",
            r"/kaggle/input/nasa-amazon-lidar-2008-2018/Nasa_lidar_2008_to_2018_DTMs/DTM_tiles/BA3_A01_2014_laz_9.tif",
        ],
        'sat_30m_dry': r"/kaggle/input/weeewooo-new-satellite-dataset/KAS_BA3_A01_2014_S2_DrySeason_50km.tif",
        'sat_30m_wet': r'/kaggle/input/weeewooo-new-satellite-dataset/KAS_BA3_A01_2014_S2_WetSeason_50km.tif',
        'hydro_dem': r"/kaggle/input/ultra-final-hydrological-dataset/hydro_extracts/BA3_A01_2014_conditioning_clipped.tif",
        'hydro_flow_dir': r"/kaggle/input/ultra-final-hydrological-dataset/hydro_extracts/BA3_A01_2014_flow_direction_clipped.tif",
        'hydro_flow_acc': r"/kaggle/input/ultra-final-hydrological-dataset/hydro_extracts/BA3_A01_2014_flow_accumulation_clipped.tif"
    },
    'BR_MT_01': { # Uses BA3_A02_2014_laz_ series
        'dtm_tile_paths_list': [ # Explicitly list all DTM tiles for this series
            r"/kaggle/input/nasa-amazon-lidar-2008-2018/Nasa_lidar_2008_to_2018_DTMs/DTM_tiles/BA3_A02_2014_laz_0.tif",
            r"/kaggle/input/nasa-amazon-lidar-2008-2018/Nasa_lidar_2008_to_2018_DTMs/DTM_tiles/BA3_A02_2014_laz_1.tif",
            r"/kaggle/input/nasa-amazon-lidar-2008-2018/Nasa_lidar_2008_to_2018_DTMs/DTM_tiles/BA3_A02_2014_laz_10.tif",
            r"/kaggle/input/nasa-amazon-lidar-2008-2018/Nasa_lidar_2008_to_2018_DTMs/DTM_tiles/BA3_A02_2014_laz_11.tif",
            r"/kaggle/input/nasa-amazon-lidar-2008-2018/Nasa_lidar_2008_to_2018_DTMs/DTM_tiles/BA3_A02_2014_laz_12.tif",
            r"/kaggle/input/nasa-amazon-lidar-2008-2018/Nasa_lidar_2008_to_2018_DTMs/DTM_tiles/BA3_A02_2014_laz_13.tif",
            r"/kaggle/input/nasa-amazon-lidar-2008-2018/Nasa_lidar_2008_to_2018_DTMs/DTM_tiles/BA3_A02_2014_laz_14.tif",
            r"/kaggle/input/nasa-amazon-lidar-2008-2018/Nasa_lidar_2008_to_2018_DTMs/DTM_tiles/BA3_A02_2014_laz_15.tif",
            r"/kaggle/input/nasa-amazon-lidar-2008-2018/Nasa_lidar_2008_to_2018_DTMs/DTM_tiles/BA3_A02_2014_laz_2.tif",
            r"/kaggle/input/nasa-amazon-lidar-2008-2018/Nasa_lidar_2008_to_2018_DTMs/DTM_tiles/BA3_A02_2014_laz_3.tif",
            r"/kaggle/input/nasa-amazon-lidar-2008-2018/Nasa_lidar_2008_to_2018_DTMs/DTM_tiles/BA3_A02_2014_laz_4.tif",
            r"/kaggle/input/nasa-amazon-lidar-2008-2018/Nasa_lidar_2008_to_2018_DTMs/DTM_tiles/BA3_A02_2014_laz_5.tif",
            r"/kaggle/input/nasa-amazon-lidar-2008-2018/Nasa_lidar_2008_to_2018_DTMs/DTM_tiles/BA3_A02_2014_laz_6.tif",
            r"/kaggle/input/nasa-amazon-lidar-2008-2018/Nasa_lidar_2008_to_2018_DTMs/DTM_tiles/BA3_A02_2014_laz_7.tif",
            r"/kaggle/input/nasa-amazon-lidar-2008-2018/Nasa_lidar_2008_to_2018_DTMs/DTM_tiles/BA3_A02_2014_laz_8.tif",
            r"/kaggle/input/nasa-amazon-lidar-2008-2018/Nasa_lidar_2008_to_2018_DTMs/DTM_tiles/BA3_A02_2014_laz_9.tif",
        ],
        'sat_30m_dry': r"/kaggle/input/weeewooo-new-satellite-dataset/KAS_BA3_A02_2014_S2_DrySeason_50km.tif",
        'sat_30m_wet': r"/kaggle/input/weeewooo-new-satellite-dataset/KAS_BA3_A02_2014_S2_WetSeason_50km.tif",
        'hydro_dem': r"/kaggle/input/ultra-final-hydrological-dataset/hydro_extracts/BA3_A02_2014_conditioning_clipped.tif",
        'hydro_flow_dir': r"/kaggle/input/ultra-final-hydrological-dataset/hydro_extracts/BA3_A02_2014_flow_direction_clipped.tif",
        'hydro_flow_acc': r"/kaggle/input/ultra-final-hydrological-dataset/hydro_extracts/BA3_A02_2014_flow_accumulation_clipped.tif"
    },
    'BR_AM_03': { # Uses DUC_A01_2012_laz_ series
        'dtm_tile_paths_list': [ # Explicitly list all DTM tiles for this series
            r"/kaggle/input/nasa-amazon-lidar-2008-2018/Nasa_lidar_2008_to_2018_DTMs/DTM_tiles/DUC_A01_2012_laz_0.tif",
            r"/kaggle/input/nasa-amazon-lidar-2008-2018/Nasa_lidar_2008_to_2018_DTMs/DTM_tiles/DUC_A01_2012_laz_1.tif",
            r"/kaggle/input/nasa-amazon-lidar-2008-2018/Nasa_lidar_2008_to_2018_DTMs/DTM_tiles/DUC_A01_2012_laz_10.tif",
            r"/kaggle/input/nasa-amazon-lidar-2008-2018/Nasa_lidar_2008_to_2018_DTMs/DTM_tiles/DUC_A01_2012_laz_11.tif",
            r"/kaggle/input/nasa-amazon-lidar-2008-2018/Nasa_lidar_2008_to_2018_DTMs/DTM_tiles/DUC_A01_2012_laz_12.tif",
            r"/kaggle/input/nasa-amazon-lidar-2008-2018/Nasa_lidar_2008_to_2018_DTMs/DTM_tiles/DUC_A01_2012_laz_13.tif",
            r"/kaggle/input/nasa-amazon-lidar-2008-2018/Nasa_lidar_2008_to_2018_DTMs/DTM_tiles/DUC_A01_2012_laz_14.tif",
            r"/kaggle/input/nasa-amazon-lidar-2008-2018/Nasa_lidar_2008_to_2018_DTMs/DTM_tiles/DUC_A01_2012_laz_15.tif",
            r"/kaggle/input/nasa-amazon-lidar-2008-2018/Nasa_lidar_2008_to_2018_DTMs/DTM_tiles/DUC_A01_2012_laz_16.tif",
            r"/kaggle/input/nasa-amazon-lidar-2008-2018/Nasa_lidar_2008_to_2018_DTMs/DTM_tiles/DUC_A01_2012_laz_17.tif",
            r"/kaggle/input/nasa-amazon-lidar-2008-2018/Nasa_lidar_2008_to_2018_DTMs/DTM_tiles/DUC_A01_2012_laz_18.tif",
            r"/kaggle/input/nasa-amazon-lidar-2008-2018/Nasa_lidar_2008_to_2018_DTMs/DTM_tiles/DUC_A01_2012_laz_19.tif",
            r"/kaggle/input/nasa-amazon-lidar-2008-2018/Nasa_lidar_2008_to_2018_DTMs/DTM_tiles/DUC_A01_2012_laz_2.tif",
            r"/kaggle/input/nasa-amazon-lidar-2008-2018/Nasa_lidar_2008_to_2018_DTMs/DTM_tiles/DUC_A01_2012_laz_20.tif",
            r"/kaggle/input/nasa-amazon-lidar-2008-2018/Nasa_lidar_2008_to_2018_DTMs/DTM_tiles/DUC_A01_2012_laz_21.tif",
            r"/kaggle/input/nasa-amazon-lidar-2008-2018/Nasa_lidar_2008_to_2018_DTMs/DTM_tiles/DUC_A01_2012_laz_22.tif",
            r"/kaggle/input/nasa-amazon-lidar-2008-2018/Nasa_lidar_2008_to_2018_DTMs/DTM_tiles/DUC_A01_2012_laz_3.tif",
            r"/kaggle/input/nasa-amazon-lidar-2008-2018/Nasa_lidar_2008_to_2018_DTMs/DTM_tiles/DUC_A01_2012_laz_4.tif",
            r"/kaggle/input/nasa-amazon-lidar-2008-2018/Nasa_lidar_2008_to_2018_DTMs/DTM_tiles/DUC_A01_2012_laz_5.tif",
            r"/kaggle/input/nasa-amazon-lidar-2008-2018/Nasa_lidar_2008_to_2018_DTMs/DTM_tiles/DUC_A01_2012_laz_6.tif",
            r"/kaggle/input/nasa-amazon-lidar-2008-2018/Nasa_lidar_2008_to_2018_DTMs/DTM_tiles/DUC_A01_2012_laz_7.tif",
            r"/kaggle/input/nasa-amazon-lidar-2008-2018/Nasa_lidar_2008_to_2018_DTMs/DTM_tiles/DUC_A01_2012_laz_8.tif",
            r"/kaggle/input/nasa-amazon-lidar-2008-2018/Nasa_lidar_2008_to_2018_DTMs/DTM_tiles/DUC_A01_2012_laz_9.tif",
        ],
        'sat_30m_dry': r"/kaggle/input/weeewooo-new-satellite-dataset/KAS_DUC_A01_2012_S2_DrySeason_50km.tif",
        'sat_30m_wet': r"/kaggle/input/weeewooo-new-satellite-dataset/KAS_DUC_A01_2012_S2_WetSeason_50km.tif",
        'hydro_dem': r"/kaggle/input/ultra-final-hydrological-dataset/hydro_extracts/DUC_A01_2012_conditioning_clipped.tif",
        'hydro_flow_dir': r"/kaggle/input/ultra-final-hydrological-dataset/hydro_extracts/DUC_A01_2012_flow_direction_clipped.tif",
        'hydro_flow_acc': r"/kaggle/input/ultra-final-hydrological-dataset/hydro_extracts/DUC_A01_2012_flow_accumulation_clipped.tif"
    }
}

# --- NEW: Global cache for CRS Transformers ---
crs_transformers_cache = {}

print("Cell 1: Setup & Global Configuration Complete.")


# -----------------------------------------------------------------------------
# Helper Functions for Audio Generation (Significantly Enhanced)
# -----------------------------------------------------------------------------
def midi_to_hz(midi_note):
    return 440 * (2**((midi_note - 69)/12))

def generate_adsr_sine_wave(frequency, duration, amplitude, sample_rate=SAMPLE_RATE, attack=0.05, decay=0.1, sustain=0.7, release=0.1):
    total_samples = int(sample_rate * duration)
    t = np.linspace(0, duration, total_samples, endpoint=False)
    wave = amplitude * np.sin(2 * np.pi * frequency * t)
    attack_samples = int(attack * sample_rate)
    samples_after_attack = total_samples - attack_samples
    decay_samples = int(decay * sample_rate)
    release_samples = int(release * sample_rate)
    sustain_samples = total_samples - attack_samples - decay_samples - release_samples

    # CORRECTED: Multi-line if/else for clarity and to fix SyntaxError
    if sustain_samples < 0:
        sustain_samples = 0
        release_samples = max(0, total_samples - attack_samples - decay_samples) # Ensure release_samples is not negative
        # If still too short for attack+decay, scale them down
        if attack_samples + decay_samples > total_samples:
            attack_samples = int(total_samples * 0.5)
            decay_samples = total_samples - attack_samples
            release_samples = 0
            sustain_samples = 0

    envelope = np.zeros(total_samples)
    if attack_samples > 0: envelope[:attack_samples] = np.linspace(0, 1, attack_samples)
    else: envelope[:1] = 1.0

    if decay_samples > 0: envelope[attack_samples : attack_samples + decay_samples] = np.linspace(1, sustain, decay_samples)
    else: envelope[attack_samples : min(attack_samples + 1, total_samples)] = sustain # Instant decay, ensure bounds

    envelope[attack_samples + decay_samples : attack_samples + decay_samples + sustain_samples] = sustain

    if release_samples > 0: envelope[total_samples - release_samples:] = np.linspace(sustain, 0, release_samples)
    else: envelope[-1:] = 0.0

    return (wave * envelope).astype(np.float32)

def generate_noise_wave(duration, amplitude, sample_rate=SAMPLE_RATE):
    return (np.random.rand(int(sample_rate * duration)) * 2 - 1) * amplitude

def butter_lowpass(cutoff, fs, order=5):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def generate_filtered_noise(duration, amplitude, cutoff_freq, sample_rate=SAMPLE_RATE, order=4):
    noise = generate_noise_wave(duration, amplitude, sample_rate)
    if cutoff_freq <= 0 or cutoff_freq >= sample_rate / 2: return noise
    b, a = butter_lowpass(cutoff_freq, sample_rate, order=order)
    filtered_noise = lfilter(b, a, noise)
    if np.max(np.abs(filtered_noise)) > 1e-6:
        filtered_noise = filtered_noise * (amplitude / np.max(np.abs(filtered_noise)))
    return filtered_noise.astype(np.float32)

def generate_glissando(start_midi, end_midi, duration, amplitude, sample_rate=SAMPLE_RATE, attack=0.05, release=0.1):
    start_freq = midi_to_hz(start_midi)
    end_freq = midi_to_hz(end_midi)
    total_samples = int(sample_rate * duration)
    t = np.linspace(0, duration, total_samples, endpoint=False)
    frequencies = np.exp(np.linspace(np.log(start_freq), np.log(end_freq), total_samples))
    wave = amplitude * np.sin(2 * np.pi * frequencies * t)
    attack_samples = int(attack * sample_rate)
    release_samples = int(release * sample_rate)
    envelope = np.ones(total_samples)
    envelope[:attack_samples] *= np.linspace(0, 1, attack_samples)
    envelope[total_samples - release_samples:] *= np.linspace(1, 0, release_samples)
    return (wave * envelope).astype(np.float32)

def generate_click(duration, amplitude, sample_rate=SAMPLE_RATE):
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    click_wave = amplitude * np.exp(-500 * t) * np.sin(2 * np.pi * 5000 * t)
    return click_wave.astype(np.float32)

def generate_pulse(bpm, duration, amplitude, sample_rate=SAMPLE_RATE, click_duration=0.05):
    bpm_scalar = bpm.item() if isinstance(bpm, np.ndarray) and bpm.size == 1 else float(bpm)
    if bpm_scalar <= 0: return np.zeros(int(sample_rate * duration), dtype=np.float32)
    samples_per_beat = sample_rate * 60 / bpm_scalar
    total_samples = int(sample_rate * duration)
    pulse_wave = np.zeros(total_samples, dtype=np.float32)
    num_beats = int(bpm_scalar * duration / 60)
    if num_beats == 0 and duration > 0: num_beats = 1
    for i in range(num_beats):
        start_idx = int(i * samples_per_beat)
        if start_idx < total_samples:
            current_click_duration = min(click_duration, duration - (start_idx / sample_rate))
            if current_click_duration > 0:
                click = generate_click(current_click_duration, amplitude, sample_rate)
                end_idx = start_idx + len(click)
                if end_idx <= total_samples: pulse_wave[start_idx:end_idx] += click
                else: pulse_wave[start_idx:total_samples] += click[:total_samples - start_idx]
    return pulse_wave

def generate_chord(root_midi, scale_midi, duration, amplitude, sample_rate=SAMPLE_RATE, chord_intervals=[0, 4, 7]):
    chord_wave = np.zeros(int(sample_rate * duration), dtype=np.float32)
    for interval in chord_intervals:
        note_freq = midi_to_hz(root_midi + interval)
        detune_factor = 1 + (np.random.rand() - 0.5) * 0.005
        chord_wave += generate_adsr_sine_wave(note_freq * detune_factor, duration, amplitude / len(chord_intervals), sample_rate, attack=0.5, decay=0.8, sustain=0.4, release=0.5)
    return chord_wave

def calculate_slope(dem_array, resolution):
    if dem_array.size == 0 or np.all(np.isnan(dem_array)): return 0.0
    if dem_array.ndim < 2 or dem_array.shape[0] < 2 or dem_array.shape[1] < 2:
        return 0.0
    dx, dy = np.gradient(dem_array.astype(float))
    if resolution == 0: return 0.0
    slope_rad = np.arctan(np.sqrt(dx**2 + dy**2) / resolution)
    return np.nanmean(np.degrees(slope_rad))

def calculate_roughness(dem_array, window_size_pixels=3):
    if dem_array.size == 0 or np.all(np.isnan(dem_array)): return 0.0
    if dem_array.ndim < 2 or dem_array.shape[0] < window_size_pixels or dem_array.shape[1] < window_size_pixels:
        return 0.0
    mean_filtered = uniform_filter(dem_array.astype(float), size=window_size_pixels, mode='reflect')
    diff = dem_array.astype(float) - mean_filtered
    if np.all(np.isnan(diff)): return 0.0
    roughness = np.nanstd(diff)
    return float(roughness)

def convert_float_to_int16(audio_array_float):
    # Ensure clipping to avoid overflow before converting to int16
    return (np.clip(audio_array_float, -1.0, 1.0) * 32767).astype(np.int16)

def get_nan_percentage(arr):
    if np.isscalar(arr): return 100.0 if np.isnan(arr) else 0.0
    if arr is None or arr.size == 0: return 100.0
    nan_count = np.sum(np.isnan(arr))
    return (nan_count / arr.size) * 100

def calculate_ndwi_s2(satellite_data_cell):
    if satellite_data_cell is None or satellite_data_cell.ndim < 3 or satellite_data_cell.shape[0] < 8:
        return np.nan
    green_band = satellite_data_cell[2, :, :]
    nir_band = satellite_data_cell[6, :, :]
    denominator = (nir_band + green_band)
    ndwi = np.where(denominator != 0, (green_band - nir_band) / denominator, np.nan)
    return np.nanmean(ndwi) if not np.all(np.isnan(ndwi)) else np.nan

# -----------------------------------------------------------------------------
# Visualization Function
# -----------------------------------------------------------------------------
def visualize_geospatial_data(transect_id, data_rasters, output_dir, transformer_transects, gan_transects):
    print(f"\n--- Generating visualizations for {transect_id} ---")
    file_suffix = ""
    if transect_id in transformer_transects: file_suffix = "_Archaeological"
    elif transect_id in gan_transects: file_suffix = "_Jungle"
    dtm_data = data_rasters.get('dtm', None)
    sat_dry_data = data_rasters.get('sat_30m_dry', None)
    hydro_flow_acc_data = data_rasters.get('hydro_flow_acc', None)
    hydro_flow_dir_data = data_rasters.get('hydro_flow_dir', None)
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    fig.suptitle(f"Geospatial Data for Transect: {transect_id}", fontsize=16)
    if dtm_data is not None and dtm_data.size > 0 and not np.all(np.isnan(dtm_data)):
        im0 = axes[0].imshow(dtm_data, cmap='terrain', origin='upper')
        axes[0].set_title('DTM (Elevation)'); fig.colorbar(im0, ax=axes[0], label='Elevation (m)')
    else: axes[0].text(0.5, 0.5, 'DTM Data Missing/Invalid', horizontalalignment='center', verticalalignment='center', transform=axes[0].transAxes, color='red'); axes[0].set_title('DTM (Elevation)')
    ndvi_data = None
    if sat_dry_data is not None and sat_dry_data.ndim > 2 and sat_dry_data.shape[0] >= 10:
        try:
            # Assuming Sentinel-2 data with 11 bands: Red (B4) is index 3, NIR (B8) is index 7
            if sat_dry_data.shape[0] >= 11:
                red_band = sat_dry_data[3, :, :]; nir_band = sat_dry_data[7, :, :]
                denominator = (nir_band + red_band)
                ndvi_data = np.where(denominator != 0, (nir_band - red_band) / denominator, np.nan)
            # If it's a pre-calculated single-band NDVI, assume it's the first band
            elif sat_dry_data.shape[0] == 1: ndvi_data = sat_dry_data[0, :, :]
            else: ndvi_data = np.nanmean(sat_dry_data[0,:,:]) # Fallback if single band and not NDVI
        except IndexError: print(f"Warning: Could not extract NDVI from sat_30m_dry for {transect_id}. Check band indexing."); ndvi_data = None
    if ndvi_data is not None and ndvi_data.size > 0 and not np.all(np.isnan(ndvi_data)):
        im1 = axes[1].imshow(ndvi_data, cmap='RdYlGn', origin='upper', vmin=-1, vmax=1)
        axes[1].set_title('NDVI (Vegetation Index)'); fig.colorbar(im1, ax=axes[1], label='NDVI Value')
    else: axes[1].text(0.5, 0.5, 'Satellite/NDVI Data Missing/Invalid', horizontalalignment='center', verticalalignment='center', transform=axes[0].transAxes, color='red'); axes[1].set_title('NDVI (Vegetation Index)')
    if hydro_flow_acc_data is not None and hydro_flow_acc_data.size > 0 and not np.all(np.isnan(hydro_flow_acc_data)):
        im2 = axes[2].imshow(np.log1p(hydro_flow_acc_data), cmap='Blues', origin='upper')
        axes[2].set_title('Log Flow Accumulation'); fig.colorbar(im2, ax=axes[2], label='Log Flow Accumulation')
    else: axes[2].text(0.5, 0.5, 'Hydro Flow Acc. Data Missing/Invalid', horizontalalignment='center', verticalalignment='center', transform=axes[2].transAxes, color='red'); axes[2].set_title('Log Flow Accumulation')
    if hydro_flow_dir_data is not None and hydro_flow_dir_data.size > 0 and not np.all(np.isnan(hydro_flow_dir_data)):
        im3 = axes[3].imshow(hydro_flow_dir_data, cmap='twilight', origin='upper')
        axes[3].set_title('Flow Direction'); fig.colorbar(im3, ax=axes[3], label='Direction Value')
    else: axes[3].text(0.5, 0.5, 'Hydro Flow Dir. Data Missing/Invalid', horizontalalignment='center', verticalalignment='center', transform=axes[3].transAxes, color='red'); axes[3].set_title('Flow Direction')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]); plt.savefig(os.path.join(output_dir, f"{transect_id}_geospatial_viz{file_suffix}.png")); plt.close(fig)
    print(f"    Visualizations saved for {transect_id} to '{output_dir}'.")

# --- NEW: Global cache for CRS Transformers ---
# This cache stores pyproj.Transformer objects to avoid redundant creation
# for repeated CRS transformations.
crs_transformers_cache = {}

# -----------------------------------------------------------------------------
# Helper Functions for data alignment and extraction (Moved from Main Loop for clarity)
# -----------------------------------------------------------------------------
def get_aligned_cell(raster_data, raster_profile, master_profile, row_idx, col_idx, pixels_per_grid_cell_master, processing_grid_size_meters):
    # DEBUG_MODE = True # Uncomment to enable debug prints for this function
    if False: # Use False by default unless you need to debug alignment
        if raster_profile and raster_profile.get('crs'):
            print(f"  Debug get_aligned_cell: Processing layer (CRS: {raster_profile['crs']}) for cell ({row_idx},{col_idx})")
        else:
            print(f"  Debug get_aligned_cell: Processing layer (No CRS info or data) for cell ({row_idx},{col_idx}). Returning empty.")

    if raster_data is None or raster_profile is None or raster_profile.get('crs') is None:
        if False: print(f"  Debug: get_aligned_cell - raster_data, raster_profile or CRS is None for cell ({row_idx},{col_idx}). Returning empty array.")
        return np.array([])

    master_transform = master_profile['transform']
    master_min_x, master_max_y = master_transform * (col_idx, row_idx)
    master_max_x, master_min_y = master_transform * (col_idx + pixels_per_grid_cell_master, row_idx + pixels_per_grid_cell_master)
    if False: print(f"  Debug: Master Cell Bounds (Master CRS {master_profile['crs']}): min_x={master_min_x:.2f}, min_y={master_min_y:.2f}, max_x={master_max_x:.2f}, max_y={master_max_y:.2f}")

    transformer_key = (str(master_profile['crs']), str(raster_profile['crs']))
    if transformer_key not in crs_transformers_cache:
        try: crs_transformers_cache[transformer_key] = Transformer.from_crs(master_profile['crs'], raster_profile['crs'], always_xy=True)
        except Exception as e: print(f"Warning: Failed to create CRS transformer from {master_profile['crs']} to {raster_profile['crs']}: {e}. Returning empty array."); return np.array([])
    transformer = crs_transformers_cache[transformer_key]

    try:
        transformed_bounds = transformer.transform_bounds(master_min_x, master_min_y, master_max_x, master_max_y)
        if False: print(f"  Debug: Transformed Cell Bounds (Target CRS {raster_profile['crs']}): {transformed_bounds}")
    except Exception as e:
        print(f"Warning: Error transforming bounds for cell ({row_idx},{col_idx}) from {master_profile['crs']} to {raster_profile['crs']}: {e}. Returning empty array.")
        return np.array([])

    try:
        window = rasterio.windows.from_bounds(*transformed_bounds, transform=raster_profile['transform'])
        window = window.round_offsets().round_lengths()

        rows_limit = raster_data.shape[1] if raster_data.ndim > 2 else raster_data.shape[0]
        cols_limit = raster_data.shape[2] if raster_data.ndim > 2 else raster_data.shape[1]

        window_col_start = max(0, window.col_off)
        window_row_start = max(0, window.row_off)
        window_col_end = min(cols_limit, window.col_off + window.width)
        window_row_end = min(rows_limit, window.row_off + window.height)

        if False:
            print(f"  Debug: Target Raster Shape: ({rows_limit}, {cols_limit})")
            print(f"  Debug: Clamped Window Coords: ({window_col_start}:{window_col_end}, {window_row_start}:{window_row_end})") # Corrected order for consistency

        if window_col_start >= window_col_end or window_row_start >= window_row_end:
            if False: print(f"  Debug: Clamped window is empty/invalid for cell ({row_idx},{col_idx}) for {raster_profile.get('crs', 'Unknown CRS')}. Returning empty array.")
            return np.array([])

        if raster_data.ndim > 2: extracted_data = raster_data[:, window_row_start:window_row_end, window_col_start:window_col_end]
        else: extracted_data = raster_data[window_row_start:window_row_end, window_col_start:window_col_end]

        if extracted_data.size == 0 or np.all(np.isnan(extracted_data)):
            if False: print(f"  Debug: Extracted data is empty or all NaNs for cell ({row_idx},{col_idx}) from {raster_profile.get('crs', 'Unknown CRS')}. Returning empty array.")
            return np.array([])

        if False: print(f"  Debug: Successfully extracted data shape: {extracted_data.shape}, is_nan: {np.any(np.isnan(extracted_data))}, min:{np.nanmin(extracted_data)}, max:{np.nanmax(extracted_data)}")
        return extracted_data

    except rasterio.errors.WindowError as e:
        if False: print(f"  Debug: WindowError for cell ({row_idx},{col_idx}) in {raster_profile.get('crs', 'Unknown CRS')}: {e}. Returning np.array([]).")
        return np.array([])
    except Exception as e:
        print(f"Warning: Error extracting data for cell ({row_idx},{col_idx}) from {raster_profile.get('crs', 'Unknown CRS')}: {e}. Returning np.array([]).")
        return np.array([])

# Define generate_rich_pulse (placeholder if not defined elsewhere)
def generate_rich_pulse(bpm, duration, amplitude, base_click_freq, sample_rate=SAMPLE_RATE, num_harmonics=3):
    total_samples = int(sample_rate * duration)
    pulse_wave = np.zeros(total_samples, dtype=np.float32)

    if bpm <= 0: return pulse_wave

    beats_per_second = bpm / 60.0
    time_between_clicks = 1.0 / beats_per_second

    num_clicks = int(duration * beats_per_second)
    if num_clicks == 0 and duration > 0:
        num_clicks = 1 # Ensure at least one click for very short durations

    for i in range(num_clicks):
        click_start_time = i * time_between_clicks
        if click_start_time >= duration:
            break

        # Generate a click with multiple sine waves for a richer sound
        click_duration_seg = min(0.05, duration - click_start_time) # Ensure click doesn't go beyond segment
        if click_duration_seg <= 0: continue

        click_segment = np.zeros(int(click_duration_seg * sample_rate), dtype=np.float32)
        for h in range(num_harmonics):
            freq = base_click_freq * (h + 1)
            # Apply a decaying envelope to each harmonic
            t_click = np.linspace(0, click_duration_seg, len(click_segment), endpoint=False)
            envelope = np.exp(-15 * t_click / click_duration_seg) # Faster decay
            click_segment += amplitude / num_harmonics * np.sin(2 * np.pi * freq * t_click) * envelope

        start_sample = int(click_start_time * sample_rate)
        end_sample = start_sample + len(click_segment)

        if end_sample <= total_samples:
            pulse_wave[start_sample:end_sample] += click_segment
        else:
            pulse_wave[start_sample:total_samples] += click_segment[:total_samples - start_sample]

    return pulse_wave

# -----------------------------------------------------------------------------
# Main Processing Loop
# -----------------------------------------------------------------------------

# Define ARCHAEOLOGICAL_TRANSECTS and JUNGLE_TRANSECTS (already defined globally at the top of the cell)
DEBUG_MODE = False

for current_transect_id in TRANSECTS_TO_PROCESS:
    print(f"\n--- Processing Transect: {current_transect_id} ---")

    current_scenario_files = TRANSECT_FILE_PATHS.get(current_transect_id)
    if not current_scenario_files:
        print(f"ERROR: File paths not defined for transect '{current_transect_id}'. Skipping.")
        continue

    output_audio_current_transect_dir = os.path.join(output_audio_base_dir, current_transect_id)
    os.makedirs(output_audio_current_transect_dir, exist_ok=True)

    # Create a temporary directory for audio chunks within the transect's output folder
    temp_audio_chunks_dir = os.path.join(output_audio_current_transect_dir, "temp_audio_chunks")
    os.makedirs(temp_audio_chunks_dir, exist_ok=True)

    data_rasters = {}
    src_profiles = {}

    # --- Load DTMs (Mosaicking Logic) ---
    # Use the explicit list of DTM tile paths from TRANSECT_FILE_PATHS
    dtm_tile_paths = current_scenario_files.get('dtm_tile_paths_list')

    if not dtm_tile_paths:
        print(f"ERROR: No 'dtm_tile_paths_list' defined for transect '{current_transect_id}'. Skipping.")
        # Ensure cleanup if we skip early
        if os.path.exists(temp_audio_chunks_dir):
            shutil.rmtree(temp_audio_chunks_dir)
        continue

    # Verify that all listed DTM files actually exist
    missing_dtm_files = [p for p in dtm_tile_paths if not os.path.exists(p)]
    if missing_dtm_files:
        print(f"ERROR: Some DTM tiles are missing for transect '{current_transect_id}': {missing_dtm_files}. Skipping.")
        # Ensure cleanup if we skip early
        if os.path.exists(temp_audio_chunks_dir):
            shutil.rmtree(temp_audio_chunks_dir)
        continue

    # Mosaic DTM tiles
    try:
        dtm_mosaic_data, dtm_mosaic_transform = rasterio.merge.merge(dtm_tile_paths)

        with rasterio.open(dtm_tile_paths[0]) as src_first_tile:
            master_profile = src_first_tile.profile.copy()

        master_profile.update({
            "height": dtm_mosaic_data.shape[1],
            "width": dtm_mosaic_data.shape[2],
            "transform": dtm_mosaic_transform,
            "nodata": master_profile.get('nodata'),
            "count": dtm_mosaic_data.shape[0] # Should be 1 for single-band DTM, but merge returns (bands, height, width)
        })
        # If dtm_mosaic_data is 3D (e.g., (bands, rows, cols)), take the first band
        data_rasters['dtm'] = dtm_mosaic_data[0] if dtm_mosaic_data.ndim == 3 else dtm_mosaic_data
        src_profiles['dtm'] = master_profile

        # CORRECTED: Get bounds from the updated profile's transform, height, and width
        merged_dtm_bounds = array_bounds(master_profile['height'], master_profile['width'], master_profile['transform'])
        print(f"    Mosaicked DTM for {current_transect_id}: Shape {data_rasters['dtm'].shape}, Bounds {merged_dtm_bounds}")
    except Exception as e:
        print(f"ERROR: Could not mosaic DTM tiles for '{current_transect_id}': {e}. Skipping transect.")
        data_rasters['dtm'] = None
        # Ensure cleanup if we skip early
        if os.path.exists(temp_audio_chunks_dir):
            shutil.rmtree(temp_audio_chunks_dir)
        continue

    # Check if DTM was successfully loaded/mosaicked
    if 'dtm' not in data_rasters or data_rasters['dtm'] is None or data_rasters['dtm'].size == 0:
        print(f"ERROR: DTM data (mosaic) not loaded for '{current_transect_id}'. Skipping sonification for this transect.")
        # Ensure cleanup if we skip early
        if os.path.exists(temp_audio_chunks_dir):
            shutil.rmtree(temp_audio_chunks_dir)
        continue


    # --- Load other rasters (Satellite & Hydro) ---
    loaded_other_rasters_successfully = True # Flag for these layers
    for key in ['sat_30m_dry', 'sat_30m_wet', 'hydro_dem', 'hydro_flow_dir', 'hydro_flow_acc']:
        path = current_scenario_files.get(key)
        if not path:
            print(f"WARNING: Path for '{key}' not defined for '{current_transect_id}'. Skipping this layer.")
            data_rasters[key] = None; src_profiles[key] = None
            continue
        if not os.path.exists(path):
            print(f"WARNING: File '{key}' not found at '{path}'. Skipping this layer for '{current_transect_id}'.")
            data_rasters[key] = None; src_profiles[key] = None
            continue
        try:
            src = rasterio.open(path)
            # Read data, casting to float32 for hydro if needed, reading all bands for sat, or first band otherwise
            if key.startswith('hydro_') and src.profile['dtype'] not in ['float32', 'float64']:
                data_rasters[key] = src.read(1, out_dtype=np.float32)
            elif key.startswith('sat_') and src.count > 1:
                data_rasters[key] = src.read() # Read all bands for satellite imagery
            else:
                data_rasters[key] = src.read(1) # Read only the first band for single-band rasters

            # Handle nodata values
            if src.nodata is not None:
                nodata_val = src.nodata
                if data_rasters[key].ndim == 3: # Multi-band case
                    for band_idx in range(data_rasters[key].shape[0]):
                        # Only replace if the band itself is not entirely NaNs (might happen if clipped to an area outside data)
                        if not np.all(np.isnan(data_rasters[key][band_idx])):
                            data_rasters[key][band_idx][data_rasters[key][band_idx] == nodata_val] = np.nan
                elif data_rasters[key].ndim == 2: # Single-band case
                    if not np.all(np.isnan(data_rasters[key])):
                        data_rasters[key][data_rasters[key] == nodata_val] = np.nan
            src_profiles[key] = src.profile
            src.close()
            print(f"    Loaded {key}: Shape {data_rasters[key].shape}, Bounds {src.bounds}")
        except Exception as e:
            print(f"ERROR loading {key} from '{path}': {e}. Skipping this layer.")
            data_rasters[key] = None; src_profiles[key] = None
            loaded_other_rasters_successfully = False # Mark as failed if loading fails

    # Check if essential satellite and hydro data were loaded
    # Assuming 'sat_30m_dry' and 'hydro_flow_acc' are critical. Adjust if other layers are critical.
    if data_rasters.get('sat_30m_dry') is None or data_rasters.get('hydro_flow_acc') is None:
        print(f"ERROR: Essential satellite (sat_30m_dry) or hydrological (hydro_flow_acc) data missing for '{current_transect_id}'. Skipping sonification for this transect.")
        # Ensure cleanup if we skip early
        if os.path.exists(temp_audio_chunks_dir):
            shutil.rmtree(temp_audio_chunks_dir)
        continue


    # --- Visualize the loaded geospatial data for the current transect ---
    visualize_geospatial_data(current_transect_id, data_rasters, output_viz_base_dir, ARCHAEOLOGICAL_TRANSECTS, JUNGLE_TRANSECTS)

    master_res = master_profile['transform'].a # Resolution of the MOSAIC DTM in its CRS
    pixels_per_grid_cell = int(PROCESSING_GRID_SIZE_METERS / master_res)
    if pixels_per_grid_cell == 0:
        print(f"ERROR: Processing grid size ({PROCESSING_GRID_SIZE_METERS}m) is smaller than or equal to master raster resolution ({master_res}m). Adjust PROCESSING_GRID_SIZE_METERS. Skipping transect.")
        # Ensure cleanup if we skip early
        if os.path.exists(temp_audio_chunks_dir):
            shutil.rmtree(temp_audio_chunks_dir)
        continue

    print(f"Using a sonification grid of approx {PROCESSING_GRID_SIZE_METERS}m x {PROCESSING_GRID_SIZE_METERS}m per audio segment for '{current_transect_id}'.")
    print("---------------------------------------------------------------")

    num_sonified_cells = 0
    total_rows, total_cols = data_rasters['dtm'].shape # Shape of the MOSAIC DTM

    print(f"Starting sonification for {total_rows // pixels_per_grid_cell} rows x {(total_cols + pixels_per_grid_cell - 1) // pixels_per_grid_cell} cols potential sonification cells...")

    temp_audio_file_paths = [] # List to store paths of temporary audio files for each cell
    cell_geometries = []
    current_audio_duration_ms = 0 # Tracks cumulative duration for CellGeom timing

    # Helper for ensuring consistent array lengths for mixing
    def ensure_length(arr, target_len):
        if len(arr) < target_len:
            return np.pad(arr, (0, target_len - len(arr)), 'constant')
        elif len(arr) > target_len:
            return arr[:target_len]
        return arr

    for row_idx in range(0, total_rows, pixels_per_grid_cell):
        for col_idx in range(0, total_cols, pixels_per_grid_cell):
            row_end = min(row_idx + pixels_per_grid_cell, total_rows)
            col_end = min(col_idx + pixels_per_grid_cell, total_cols)

            dtm_cell = data_rasters['dtm'][row_idx:row_end, col_idx:col_end]

            sat_dry_cell_data = get_aligned_cell(data_rasters.get('sat_30m_dry'), src_profiles.get('sat_30m_dry'), master_profile, row_idx, col_idx, pixels_per_grid_cell, PROCESSING_GRID_SIZE_METERS)
            sat_wet_cell_data = get_aligned_cell(data_rasters.get('sat_30m_wet'), src_profiles.get('sat_30m_wet'), master_profile, row_idx, col_idx, pixels_per_grid_cell, PROCESSING_GRID_SIZE_METERS)

            hydro_dem_cell = get_aligned_cell(data_rasters.get('hydro_dem'), src_profiles.get('hydro_dem'), master_profile, row_idx, col_idx, pixels_per_grid_cell, PROCESSING_GRID_SIZE_METERS)
            hydro_flow_dir_cell = get_aligned_cell(data_rasters.get('hydro_flow_dir'), src_profiles.get('hydro_flow_dir'), master_profile, row_idx, col_idx, pixels_per_grid_cell, PROCESSING_GRID_SIZE_METERS)
            hydro_flow_acc_cell = get_aligned_cell(data_rasters.get('hydro_flow_acc'), src_profiles.get('hydro_flow_acc'), master_profile, row_idx, col_idx, pixels_per_grid_cell, PROCESSING_GRID_SIZE_METERS)

            ndvi_cell_array = np.nan; evi_cell_array = np.nan; bsi_cell_array = np.nan
            ndvi_cell = np.nan; evi_cell = np.nan; bsi_cell = np.nan; brightness_cell = np.nan

            if sat_dry_cell_data.size > 0 and sat_dry_cell_data.ndim == 3 and sat_dry_cell_data.shape[0] >= 11:
                # Assuming Sentinel-2 bands: Blue(B2)=0, Green(B3)=1, Red(B4)=2, VEG_RED(B5)=3, VEG_RED(B6)=4, VEG_RED(B7)=5, NIR(B8)=6, NIR(B8A)=7, SWIR1(B11)=9, SWIR2(B12)=10
                # Correct indices for Sentinel-2 bands used in calculations
                red_band = sat_dry_cell_data[3, :, :]; nir_band = sat_dry_cell_data[7, :, :]; blue_band = sat_dry_cell_data[0, :, :]; green_band = sat_dry_cell_data[1, :, :]; swir1_band = sat_dry_cell_data[9, :, :]
                denominator_ndvi = nir_band + red_band
                ndvi_cell_array = np.where(denominator_ndvi != 0, (nir_band - red_band) / denominator_ndvi, np.nan); ndvi_cell = np.nanmean(ndvi_cell_array)
                denominator_evi = nir_band + 6 * red_band - 7.5 * blue_band + 1
                evi_cell_array = np.where(denominator_evi != 0, 2.5 * ((nir_band - red_band) / denominator_evi), np.nan); evi_cell = np.nanmean(evi_cell_array)
                numerator_bsi = (swir1_band + red_band) - (nir_band + blue_band); denominator_bsi = (swir1_band + red_band) + (nir_band + blue_band)
                bsi_cell_array = np.where(denominator_bsi != 0, numerator_bsi / denominator_bsi, np.nan); bsi_cell = np.nanmean(bsi_cell_array)
                brightness_cell = np.nanmean(swir1_band) # Using SWIR1 as a proxy for brightness
            elif sat_dry_cell_data.size > 0 and sat_dry_cell_data.ndim == 2:
                # If it's a single-band raster assumed to be NDVI or brightness already
                ndvi_cell_array = sat_dry_cell_data; ndvi_cell = np.nanmean(ndvi_cell_array)
                brightness_cell = np.nanmean(sat_dry_cell_data)

            mean_ndvi = ndvi_cell if not np.isnan(ndvi_cell) else 0.0
            mean_evi = evi_cell if not np.isnan(evi_cell) else 0.0
            mean_bsi_30m = bsi_cell if not np.isnan(bsi_cell) else 0.0
            mean_brightness = brightness_cell if not np.isnan(brightness_cell) else 0.0
            mean_ndwi = calculate_ndwi_s2(sat_dry_cell_data) # Recalculate NDWI if needed, or ensure it's handled

            minx, maxy = master_profile['transform'] * (col_idx, row_idx)
            maxx, miny = master_profile['transform'] * (col_idx + pixels_per_grid_cell, row_idx + pixels_per_grid_cell)

            dtm_nan_percent = get_nan_percentage(dtm_cell)
            ndvi_nan_percent = get_nan_percentage(ndvi_cell_array if not np.isscalar(ndvi_cell_array) else np.nan)
            hydro_flow_acc_nan_percent = get_nan_percentage(hydro_flow_acc_cell)

            print(f"    Cell R{row_idx // pixels_per_grid_cell}_C{col_idx // pixels_per_grid_cell}: "
                  f"DTM NaN={dtm_nan_percent:.1f}%, NDVI NaN={ndvi_nan_percent:.1f}%, FlowAcc NaN={hydro_flow_acc_nan_percent:.1f}%, NDWI={mean_ndwi:.2f}")

            is_valid_cell = (
                dtm_cell.size > 0 and not np.all(np.isnan(dtm_cell)) and
                not np.isnan(mean_ndvi) and # Check for valid NDVI
                hydro_flow_acc_cell.size > 0 and not np.all(np.isnan(hydro_flow_acc_cell)) # Check for valid flow acc
            )

            if not is_valid_cell:
                print(f"    !!! Cell R{row_idx // pixels_per_grid_cell}_C{col_idx // pixels_per_grid_cell} is INVALID - generating silent audio. !!!")
                mixed_cell_audio_np = np.zeros(int(SAMPLE_RATE * DURATION_PER_GRID_CELL), dtype=np.float32)

                # Export silent audio to a temporary file
                temp_audio_filename = f"cell_{row_idx}_{col_idx}.wav"
                temp_audio_path = os.path.join(temp_audio_chunks_dir, temp_audio_filename)
                sf.write(temp_audio_path, mixed_cell_audio_np, SAMPLE_RATE, subtype='PCM_16')
                temp_audio_file_paths.append(temp_audio_path)

                cell_duration_ms = DURATION_PER_GRID_CELL * 1000 # Since it's a fixed duration
                cell_geometries.append(CellGeom(minx, miny, maxx, maxy, current_audio_duration_ms, current_audio_duration_ms + cell_duration_ms))
                current_audio_duration_ms += cell_duration_ms
                continue

            num_sonified_cells += 1

            mean_elevation = np.nanmean(dtm_cell); std_dev_elevation = np.nanstd(dtm_cell)
            mean_slope = calculate_slope(dtm_cell, master_res); mean_roughness = calculate_roughness(dtm_cell)
            mean_flow_acc = np.nanmean(hydro_flow_acc_cell); mean_hydro_dem = np.nanmean(hydro_dem_cell); mean_flow_dir = np.nanmean(hydro_flow_dir_cell)


            # Sonification parameter mapping
            clipped_ndvi = np.clip(mean_ndvi, -0.2, 0.8); ndvi_normalized_for_pitch = np.interp(clipped_ndvi, [-0.2, 0.8], [0.0, 1.0])
            pitch_interpolation_factor = 1.0 - ndvi_normalized_for_pitch # Invert for lower NDVI = higher pitch
            current_base_freq_min_midi = np.interp(pitch_interpolation_factor, [0,1], [GLOBAL_NDVI_PITCH_LOW_MIN_MIDI, GLOBAL_NDVI_PITCH_HIGH_MIN_MIDI])
            current_base_freq_max_midi = np.interp(pitch_interpolation_factor, [0,1], [GLOBAL_NDVI_PITCH_LOW_MAX_MIDI, GLOBAL_NDVI_PITCH_HIGH_MAX_MIDI])
            current_base_freq_min_hz = midi_to_hz(current_base_freq_min_midi); current_base_freq_max_hz = midi_to_hz(current_base_freq_max_midi)
            current_scale = MAJOR_SCALE_MIDI if pitch_interpolation_factor > 0.5 else MINOR_PENTATONIC_MIDI
            current_roughness_filter_min = np.interp(pitch_interpolation_factor, [0,1], [GLOBAL_ROUGHNESS_FILTER_LOW_MIN, GLOBAL_ROUGHNESS_FILTER_HIGH_MIN])
            current_roughness_filter_max = np.interp(pitch_interpolation_factor, [0,1], [GLOBAL_ROUGHNESS_FILTER_LOW_MAX, GLOBAL_ROUGHNESS_FILTER_HIGH_MAX])
            current_hydro_drone_freq_min = np.interp(pitch_interpolation_factor, [0,1], [GLOBAL_HYDRO_DRONE_LOW_MIN, GLOBAL_HYDRO_DRONE_HIGH_MIN])
            current_hydro_drone_freq_max = np.interp(pitch_interpolation_factor, [0,1], [GLOBAL_HYDRO_DRONE_LOW_MAX, GLOBAL_HYDRO_DRONE_HIGH_MAX])


            # 1. Topography Bass/Drone (Elevation)
            base_freq_low = np.interp(mean_elevation, [0, 500], [current_base_freq_min_hz, current_base_freq_max_hz])
            bass_harmonics = [base_freq_low / 2, base_freq_low, base_freq_low * 1.5]
            topography_bass_wave = np.zeros(int(SAMPLE_RATE * DURATION_PER_GRID_CELL), dtype=np.float32)
            for freq in bass_harmonics:
                topography_bass_wave += generate_adsr_sine_wave(freq, DURATION_PER_GRID_CELL, 0.4 / len(bass_harmonics),
                                                                 attack=0.8, decay=1.0, sustain=0.7, release=1.0)

            filter_mod_depth = np.interp(std_dev_elevation, [0, 20], [0, 0.2]); filter_mod_rate = 0.5 + np.random.rand() * 1.5
            t_mod = np.linspace(0, DURATION_PER_GRID_CELL, len(topography_bass_wave), endpoint=False)
            topography_bass_wave *= (1 + filter_mod_depth * np.sin(2 * np.pi * filter_mod_rate * t_mod))


            # 2. Slope Percussion / Rhythmic Element
            pulse_bpm = np.interp(mean_slope, [0, 45], [60, 180]); pulse_amplitude = np.interp(mean_slope, [0, 45], [0.0, 0.4])
            click_base_freq = np.interp(mean_roughness, [0, 10], [current_roughness_filter_min, current_roughness_filter_max])
            dtm_percussion_wave = generate_rich_pulse(pulse_bpm, DURATION_PER_GRID_CELL, pulse_amplitude, click_base_freq)

            # 3. Roughness Texture (Filtered Noise)
            noise_cutoff_freq = np.interp(mean_roughness, [0, 10], [current_roughness_filter_min, current_roughness_filter_max])
            noise_amplitude = np.interp(std_dev_elevation, [0, 20], [0.0, 0.4])
            roughness_texture_wave = np.zeros(int(SAMPLE_RATE * DURATION_PER_GRID_CELL), dtype=np.float32)
            for _ in range(3): # Layer multiple noise instances for richness
                detune_cutoff = noise_cutoff_freq * (1 + (np.random.rand() - 0.5) * 0.1) # Slight detuning for depth
                roughness_texture_wave += generate_filtered_noise(DURATION_PER_GRID_CELL, noise_amplitude / 3, detune_cutoff, sample_rate=SAMPLE_RATE)

            # 4. NDVI/EVI Melody/Chord Layer
            scaled_ndvi_midi_for_melody = np.interp(ndvi_normalized_for_pitch, [0.0, 1.0], [current_base_freq_min_midi, current_base_freq_max_midi])
            # Snap to nearest note in the current scale
            closest_scale_note_midi_relative = current_scale[np.argmin(np.abs(np.array(current_scale) % 12 - (scaled_ndvi_midi_for_melody % 12)))]
            target_octave = int(scaled_ndvi_midi_for_melody // 12)
            melody_root_midi = closest_scale_note_midi_relative + target_octave * 12

            evi_chord_density = np.interp(mean_evi, [0.0, 0.8], [0, 1])
            melody_audio_array = np.zeros(int(SAMPLE_RATE * DURATION_PER_GRID_CELL), dtype=np.float32)
            chord_intervals = [0] # Always include the root
            if evi_chord_density > 0.3: chord_intervals.append(np.random.choice([3,4])); # Minor or Major third
            if evi_chord_density > 0.6: chord_intervals.append(7); # Perfect fifth
            if evi_chord_density > 0.8: chord_intervals.append(10) # Minor seventh for more complex chords

            num_chord_hits = 2; # Number of times the chord will be struck per cell
            hit_duration = DURATION_PER_GRID_CELL / num_chord_hits
            for i in range(num_chord_hits):
                hit_start_time = i * hit_duration
                chord_amplitude = np.interp(mean_ndvi, [0.0, 0.8], [0.3, 0.6])
                chord_wave = generate_chord(melody_root_midi, current_scale, hit_duration, chord_amplitude, chord_intervals=chord_intervals, sample_rate=SAMPLE_RATE)
                start_sample = int(hit_start_time * SAMPLE_RATE)
                end_sample = start_sample + len(chord_wave)
                if end_sample <= len(melody_audio_array): melody_audio_array[start_sample:end_sample] += chord_wave
                else: melody_audio_array[start_sample:] += chord_wave[:len(melody_audio_array) - start_sample]

            # 5. Hydrological Drone (Flow Accumulation and DEM)
            log_flow_acc_scaled = np.interp(np.log1p(mean_flow_acc), [0, np.log1p(100000)], [0.0, 1.0]) # Log scale for large range
            hydro_drone_freq = np.interp(mean_hydro_dem, [0, 200], [current_hydro_drone_freq_min, current_hydro_drone_freq_max])
            hydro_drone_amplitude = log_flow_acc_scaled * 0.6
            pitch_bend = (np.random.rand() - 0.5) * 0.02 # Small random pitch variation
            hydro_layer_audio_np = generate_adsr_sine_wave(hydro_drone_freq * (1 + pitch_bend), DURATION_PER_GRID_CELL, hydro_drone_amplitude, attack=1.5, decay=1.5, sustain=0.7, release=1.5, sample_rate=SAMPLE_RATE)

            # Apply gain adjustments based on water body detection (NDWI)
            gain_topo, gain_dtm_perc, gain_roughness, gain_melody, gain_hydro = 0.0, 0.0, 0.0, 0.0, -5 # Default gains
            if mean_ndwi > NDWI_WATER_THRESHOLD:
                print(f"        !!! Water body detected (NDWI={mean_ndwi:.2f}) - suppressing other layers and boosting hydro. !!!")
                gain_topo = WATER_BODY_SUPPRESSION_GAIN_DB
                gain_dtm_perc = WATER_BODY_SUPPRESSION_GAIN_DB
                gain_roughness = WATER_BODY_SUPPRESSION_GAIN_DB
                gain_melody = WATER_BODY_SUPPRESSION_GAIN_DB
                gain_hydro = HYDRO_BOOST_GAIN_DB # Boost hydro for water bodies
            else: # Default gains for non-water areas
                gain_topo, gain_dtm_perc, gain_roughness, gain_melody = -6, -12, -9, -4 # Fine-tune these for overall mix

            # Combine all audio layers into a single NumPy array for the cell
            total_cell_samples = int(SAMPLE_RATE * DURATION_PER_GRID_CELL)
            mixed_cell_audio_np = np.zeros(total_cell_samples, dtype=np.float32)

            mixed_cell_audio_np += ensure_length(topography_bass_wave, total_cell_samples) * (10**(gain_topo/20.0))
            mixed_cell_audio_np += ensure_length(dtm_percussion_wave, total_cell_samples) * (10**(gain_dtm_perc/20.0))
            mixed_cell_audio_np += ensure_length(roughness_texture_wave, total_cell_samples) * (10**(gain_roughness/20.0))
            mixed_cell_audio_np += ensure_length(melody_audio_array, total_cell_samples) * (10**(gain_melody/20.0))
            mixed_cell_audio_np += ensure_length(hydro_layer_audio_np, total_cell_samples) * (10**(gain_hydro/20.0))


            # Convert to AudioSegment for pydub's pan effect
            # We convert to int16 before passing to AudioSegment because sf.read expects it later
            mixed_cell_audio_segment = AudioSegment(convert_float_to_int16(mixed_cell_audio_np).tobytes(),
                                                    frame_rate=SAMPLE_RATE, sample_width=2, channels=1) # 2 bytes = int16

            # Panning based on flow direction (simplified for effect)
            pan_value = 0.0
            if mean_flow_dir > 0:
                # D8 flow directions: 1 (E), 2 (NE), 4 (N), 8 (NW), 16 (W), 32 (SW), 64 (S), 128 (SE)
                # Map directions to pan (-1.0 for hard left, 1.0 for hard right)
                if mean_flow_dir == 1: pan_value = 1.0 # East
                elif mean_flow_dir == 2: pan_value = 0.7 # Northeast
                elif mean_flow_dir == 4: pan_value = 0.0 # North (Center)
                elif mean_flow_dir == 8: pan_value = -0.7 # Northwest
                elif mean_flow_dir == 16: pan_value = -1.0 # West
                elif mean_flow_dir == 32: pan_value = -0.7 # Southwest
                elif mean_flow_dir == 64: pan_value = 0.0 # South (Center)
                elif mean_flow_dir == 128: pan_value = 0.7 # Southeast
            mixed_cell_audio_segment = mixed_cell_audio_segment.pan(pan_value)

            # Anomaly Detection and Sonification (Archaeological vs. Jungle)
            is_anomaly_cell = False
            # Define anomaly regions (example coordinates based on your problem description's approximate areas)
            if current_transect_id == 'BR_AC_10':
                cell_row_index = row_idx // pixels_per_grid_cell
                cell_col_index = col_idx // pixels_per_grid_cell # FIXED: Corrected to pixels_per_grid_cell
                if (cell_row_index >= 4 and cell_row_index < 9 and
                    cell_col_index >= 6 and cell_col_index < 11):
                    is_anomaly_cell = True
            elif current_transect_id == 'BR_RO_05':
                cell_row_index = row_idx // pixels_per_grid_cell
                cell_col_index = col_idx // pixels_per_grid_cell # FIXED: Corrected to pixels_per_grid_cell
                if (cell_row_index >= 10 and cell_row_index < 15 and
                    cell_col_index >= 12 and cell_col_index < 17):
                    is_anomaly_cell = True
            elif current_transect_id == 'BR_PA_02':
                cell_row_index = row_idx // pixels_per_grid_cell
                cell_col_index = col_idx // pixels_per_grid_cell # FIXED: Corrected to pixels_per_grid_cell
                if (cell_row_index >= 25 and cell_row_index < 30 and
                    cell_col_index >= 8 and cell_col_index < 13):
                    is_anomaly_cell = True
            elif current_transect_id == 'BR_AC_07':
                cell_row_index = row_idx // pixels_per_grid_cell
                cell_col_index = col_idx // pixels_per_grid_cell # FIXED: Corrected to pixels_per_grid_cell
                if (cell_row_index >= 5 and cell_row_index < 10 and
                    cell_col_index >= 5 and cell_col_index < 10):
                    is_anomaly_cell = True
            elif current_transect_id == 'BR_AC_09':
                cell_row_index = row_idx // pixels_per_grid_cell
                cell_col_index = col_idx // pixels_per_grid_cell # FIXED: Corrected to pixels_per_grid_cell
                if (cell_row_index >= 7 and cell_row_index < 12 and
                    cell_col_index >= 7 and cell_col_index < 12):
                    is_anomaly_cell = True


            if is_anomaly_cell:
                if current_transect_id in ARCHAEOLOGICAL_TRANSECTS:
                    print(f"    !!! ANOMALY TRIGGERED and PIERCING PING ADDED for {current_transect_id} at cell R{row_idx // pixels_per_grid_cell}_C{col_idx // pixels_per_grid_cell} !!!")
                    # More dramatic/alarming sound for archaeological anomalies
                    siren_gliss = generate_glissando(ANOMALY_GLISS_MIDI_START - 24, ANOMALY_GLISS_MIDI_END + 24, DURATION_PER_GRID_CELL, 0.9, attack=0.1, release=0.5, sample_rate=SAMPLE_RATE)
                    harsh_noise = generate_filtered_noise(DURATION_PER_GRID_CELL, 0.8, 15000, order=1, sample_rate=SAMPLE_RATE)
                    sub_drop = generate_adsr_sine_wave(30, DURATION_PER_GRID_CELL, 0.7, attack=0.05, decay=0.8, sustain=0.1, release=0.2, sample_rate=SAMPLE_RATE)
                    
                    # Ensure all anomaly components have the same length as DURATION_PER_GRID_CELL
                    siren_gliss_padded = ensure_length(siren_gliss, total_cell_samples)
                    harsh_noise_padded = ensure_length(harsh_noise, total_cell_samples)
                    sub_drop_padded = ensure_length(sub_drop, total_cell_samples)
                    
                    anomaly_core_np = siren_gliss_padded + harsh_noise_padded * 0.5 + sub_drop_padded * 0.7 # Combine for complex sound
                    
                    piercing_ping_wave = generate_adsr_sine_wave(midi_to_hz(ANOMALY_PING_MIDI_NOTE), ANOMALY_PING_DURATION, ANOMALY_PING_AMPLITUDE, attack=0.01, decay=0.05, sustain=0.0, release=0.1, sample_rate=SAMPLE_RATE)
                    piercing_ping_wave_padded = ensure_length(piercing_ping_wave, total_cell_samples) # Pad if needed

                    anomaly_segment = AudioSegment(convert_float_to_int16(anomaly_core_np).tobytes(), frame_rate=SAMPLE_RATE, sample_width=2, channels=1)
                    piercing_ping_segment = AudioSegment(convert_float_to_int16(piercing_ping_wave_padded).tobytes(), frame_rate=SAMPLE_RATE, sample_width=2, channels=1)

                    # Overlay ping at the start of the segment
                    anomaly_segment_final = anomaly_segment.overlay(piercing_ping_segment, position=0, gain_during_overlay=0)
                    # Overlay the anomaly segment onto the already mixed cell audio segment
                    mixed_cell_audio_segment = mixed_cell_audio_segment.overlay(anomaly_segment_final.set_frame_rate(SAMPLE_RATE), gain_during_overlay=-3) # Overlay on top of existing mix

                elif current_transect_id in JUNGLE_TRANSECTS:
                    print(f"    !!! ANOMALY TRIGGERED (Jungle type - subtle) for {current_transect_id} at cell R{row_idx // pixels_per_grid_cell}_C{col_idx // pixels_per_grid_cell} !!!")
                    # More subtle, natural-sounding anomaly for jungle
                    jungle_anomaly_sound = generate_glissando(ANOMALY_GLISS_MIDI_START - 36, ANOMALY_GLISS_MIDI_START - 24, DURATION_PER_GRID_CELL, 0.7, sample_rate=SAMPLE_RATE, attack=0.2, release=0.2)
                    jungle_anomaly_sound_padded = ensure_length(jungle_anomaly_sound, total_cell_samples)
                    jungle_anomaly_segment = AudioSegment(convert_float_to_int16(jungle_anomaly_sound_padded).tobytes(), frame_rate=SAMPLE_RATE, sample_width=2, channels=1)
                    # Overlay the anomaly segment onto the already mixed cell audio segment
                    mixed_cell_audio_segment = mixed_cell_audio_segment.overlay(jungle_anomaly_segment.set_frame_rate(SAMPLE_RATE), gain_during_overlay=-6)


            # Export the final mixed and possibly anomaly-modified segment to a temporary file
            temp_audio_filename = f"cell_{row_idx}_{col_idx}.wav"
            temp_audio_path = os.path.join(temp_audio_chunks_dir, temp_audio_filename)
            
            # Ensure the output format is compatible with int16 and pydub/soundfile
            mixed_cell_audio_segment.export(temp_audio_path, format="wav")
            temp_audio_file_paths.append(temp_audio_path)
            
            # Calculate duration for CellGeom from the pydub segment's actual duration
            cell_duration_ms = mixed_cell_audio_segment.duration_seconds * 1000
            cell_geometries.append(CellGeom(minx, miny, maxx, maxy, current_audio_duration_ms, current_audio_duration_ms + cell_duration_ms))
            current_audio_duration_ms += cell_duration_ms


    print(f"\n--- Finalizing Audio for {current_transect_id} ---")

    # Determine file suffix based on transect category for naming
    # FIXED: Ensure file_suffix is defined even if no categories match
    file_suffix = "" # Default to empty string
    if current_transect_id in ARCHAEOLOGICAL_TRANSECTS:
        file_suffix = "_Archaeological"
    elif current_transect_id in JUNGLE_TRANSECTS:
        file_suffix = "_Jungle"
    # Add an 'else' if you want a default suffix for CITY_TRANSECTS or uncategorized.
    # For now, it will be an empty string for cities/uncategorized.

    # Define a path for the intermediate concatenated WAV
    final_output_concat_path = os.path.join(output_audio_current_transect_dir, f"{current_transect_id}_raw_concatenated.wav")
    final_output_final_path = os.path.join(output_audio_current_transect_dir, f"{current_transect_id}_full_sonification_SOTA{file_suffix}.wav")


    if temp_audio_file_paths:
        try:
            # Read parameters from the first temporary file
            with sf.SoundFile(temp_audio_file_paths[0], 'r') as f_read:
                samplerate_out = f_read.samplerate
                channels_out = 1 # FIXED: Explicitly set to 1 channel (mono) for consistency
                subtype_out = f_read.subtype
                file_format_out = f_read.format

            # Use soundfile to write all chunks to a single intermediate file
            # This operation is memory-efficient as it streams data from disk to disk
            with sf.SoundFile(final_output_concat_path, 'w', samplerate_out, channels_out, subtype=subtype_out, format=file_format_out) as f_write:
                for temp_file in temp_audio_file_paths:
                    # sf.read reads data, `_` discards samplerate from tuple
                    data, _ = sf.read(temp_file, dtype='int16') # Read as int16
                    if data.ndim == 2: # Safeguard: if somehow stereo, take first channel (or average)
                        data = data[:, 0]
                    f_write.write(data) # Write to the concatenated file

            # --- Memory-Efficient Normalization Pass (replaces pydub.normalize for full file) ---
            # 1. First pass to find the peak amplitude
            max_amplitude = 0.0
            block_size = 4096 # Process in small blocks
            try:
                with sf.SoundFile(final_output_concat_path, 'r') as f_read_peak:
                    for block in f_read_peak.blocks(blocksize=block_size, dtype='float32'): # Read as float for peak finding
                        current_max = np.max(np.abs(block))
                        if current_max > max_amplitude:
                            max_amplitude = current_max
            except Exception as e:
                print(f"Warning: Could not determine peak for normalization ({e}). Proceeding without normalization.")
                max_amplitude = 1.0 # Default to no normalization if peak finding fails

            normalization_factor = 1.0 # Default if audio is silent
            if max_amplitude > 1e-6: # Avoid division by zero for silent files
                # Normalize to near 1.0 (e.g., 0.95 to avoid clipping after potential re-encoding)
                normalization_factor = 0.95 / max_amplitude

            # 2. Second pass to apply normalization and write the final output
            # No dynamic range compression here due to memory constraints and complexity.
            # If dynamic range compression is crucial, it should be done externally with specialized tools
            # or by highly optimized, streaming libraries not available out-of-the-box in pydub/soundfile.
            
            with sf.SoundFile(final_output_concat_path, 'r') as f_read_norm:
                with sf.SoundFile(final_output_final_path, 'w', samplerate_out, channels_out, subtype=subtype_out, format=file_format_out) as f_write_norm:
                    for block in f_read_norm.blocks(blocksize=block_size, dtype='float32'):
                        normalized_block = block * normalization_factor
                        f_write_norm.write(normalized_block) # sf handles float32 to int16 conversion here

            print(f"    Audio successfully concatenated and normalized to: '{final_output_final_path}'")
            print("    Note: Dynamic range compression was skipped due to memory constraints. Consider external post-processing if needed.")

        except Exception as e:
            print(f"ERROR during audio concatenation/normalization for '{current_transect_id}': {e}. Generating silent output.")
            # If anything goes wrong, ensure a silent file is still created as final output
            silent_np_array = np.zeros(int(current_audio_duration_ms / 1000 * SAMPLE_RATE), dtype=np.int16)
            sf.write(final_output_final_path, silent_np_array, SAMPLE_RATE, subtype='PCM_16')
            print(f"    Silent placeholder file generated: '{final_output_final_path}'")
            
            # Ensure the temp concat file is cleaned up if it was partially created
            if os.path.exists(final_output_concat_path):
                os.remove(final_output_concat_path)
    else:
        print(f"No valid audio chunks generated for '{current_transect_id}'. Generating silent output.")
        # If no valid audio was generated at all, create a silent file
        silent_np_array = np.zeros(int(current_audio_duration_ms / 1000 * SAMPLE_RATE), dtype=np.int16)
        sf.write(final_output_final_path, silent_np_array, SAMPLE_RATE, subtype='PCM_16')
        print(f"    Silent placeholder file generated: '{final_output_final_path}'")


    # New: Export the geospatial metadata to a JSON file
    json_output_filename = os.path.join(output_audio_current_transect_dir, f"{current_transect_id}_geospatial_metadata.json")
    with open(json_output_filename, 'w') as f:
        json.dump([cg.to_dict() for cg in cell_geometries], f, indent=4)
    print(f"    Geospatial metadata saved to: '{json_output_filename}'")

    # --- Cleanup Temporary Audio Chunks ---
    if os.path.exists(temp_audio_chunks_dir):
        shutil.rmtree(temp_audio_chunks_dir)
        print(f"    Cleaned up temporary audio chunks directory: '{temp_audio_chunks_dir}'")
    # Clean up the intermediate concatenated file
    if os.path.exists(final_output_concat_path):
        os.remove(final_output_concat_path)
        print(f"    Cleaned up intermediate concatenated file: '{final_output_concat_path}'")


    print(f"\n--- Sonification Process Complete for {current_transect_id} ---")
    print(f"Generated FULL WAV file for '{current_transect_id}'.")
    print(f"File saved to: '{final_output_final_path}'") # Use the new final path
    print("---------------------------------------------------------------")

print("\nAll selected transects processed.")
print(f"Overall output directory: '{output_audio_base_dir}'")
