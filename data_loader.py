import os
import json
import numpy as np
import rasterio
from utils.logger import log

def load_raster(path):
    try:
        with rasterio.open(path) as src:
            return src.read(1), src.transform
    except Exception as e:
        log(f"Failed to load raster: {path} | {e}", level="ERROR")
        return None, None

def load_embeddings(path):
    if not os.path.exists(path):
        log(f"Embedding file not found: {path}", level="WARN")
        return np.array([])
    return np.load(path)

def load_json(path):
    if not os.path.exists(path):
        log(f"JSON not found: {path}", level="WARN")
        return []
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception as e:
        log(f"Error loading JSON {path}: {e}", level="ERROR")
        return []

def save_json(data, path):
    try:
        with open(path, "w") as f:
            json.dump(data, f, indent=4)
        log(f"Saved JSON: {path}")
    except Exception as e:
        log(f"Failed to save JSON {path}: {e}", level="ERROR")
