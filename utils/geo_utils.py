import rasterio
from shapely.geometry import box
from pyproj import Transformer
import numpy as np

def align_rasters(base_raster_path, target_raster_path):
    """Align two rasters to the same CRS and resolution."""
    with rasterio.open(base_raster_path) as base, rasterio.open(target_raster_path) as target:
        transformer = Transformer.from_crs(target.crs, base.crs, always_xy=True)
        data = target.read(1)
        return data, transformer

def get_cell_bounds(x_min, y_min, cell_size):
    """Return bounding box coordinates for a given cell."""
    return box(x_min, y_min, x_min + cell_size, y_min + cell_size)

def calculate_slope(dtm_array, transform):
    """Calculate slope in degrees from DTM raster."""
    x, y = np.gradient(dtm_array, transform[0], transform[4])
    slope = np.degrees(np.arctan(np.sqrt(x*x + y*y)))
    return slope

def calculate_roughness(dtm_array):
    """Calculate terrain roughness."""
    return np.std(dtm_array)

def ndvi(red_band, nir_band):
    """Compute NDVI from red and NIR bands."""
    return (nir_band - red_band) / (nir_band + red_band + 1e-10)

def ndwi(green_band, nir_band):
    """Compute NDWI from green and NIR bands."""
    return (green_band - nir_band) / (green_band + nir_band + 1e-10)
