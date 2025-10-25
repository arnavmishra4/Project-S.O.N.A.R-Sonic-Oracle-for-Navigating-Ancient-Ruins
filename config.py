import os

# Base directories
BASE_DIR = os.getcwd()
SONIFIED_AUDIO_BASE_DIR = os.path.join(BASE_DIR, "data/sonified_outputs")
EMBEDDING_OUTPUT_DIR = os.path.join(BASE_DIR, "data/audio_embeddings")
ANOMALY_OUTPUT_DIR = os.path.join(BASE_DIR, "data/anomaly_results")
MOTIF_OUTPUT_DIR = os.path.join(BASE_DIR, "data/motif_recognition_results")
# LIDAR and HydroSHEDS configuration
LIDAR_DTM_TILES_DIR = "data/lidar/Nasa_lidar_2008_to_2018_DTMs/DTM_tiles"
HYDRO_GLOBAL_BASE_DIR = "data/hydrosheds"

HYDRO_GLOBAL_FILES = {
    "conditioned_dem": f"{HYDRO_GLOBAL_BASE_DIR}/sa_con_3s/sa_con_3s.tif",
    "flow_direction": f"{HYDRO_GLOBAL_BASE_DIR}/sa_dir_3s/sa_dir_3s.tif",
    "flow_accumulation": f"{HYDRO_GLOBAL_BASE_DIR}/sa_acc_3s/sa_acc_3s.tif",
}

# Sample rate and durations
AUDIO_SAMPLE_RATE = 44100
VGGISH_SAMPLE_RATE = 16000
DURATION_PER_GRID_CELL = 6.0  # seconds

# Anomaly detection
ISOLATION_FOREST_RANDOM_STATE = 42
ISOLATION_FOREST_CONTAMINATION = 0.01

# DTW
DTW_SIMILARITY_THRESHOLD = 75
