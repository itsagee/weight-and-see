import numpy as np
from PIL import Image
from . import colour_space_conversions as cs

# === Grayscale Helpers ===

# helper function to turn an image into grayscale using standard luminance weights
def to_grayscale(image: np.ndarray) -> np.ndarray:
    return 0.2126 * image[:, :, 0] + 0.7152 * image[:, :, 1] + 0.0722 * image[:, :, 2]

# === Coloured Helpers ===

# helper function to find the nearest palette colour for a given pixel
def find_nearest_colour(pixel: np.ndarray, palette: np.ndarray) -> int:
    # now we can compute the distances to all palette colours in the same way across all colour spaces
    distances = np.linalg.norm(palette - pixel, axis=1)
    # and now we can return the index of nearest colour
    return np.argmin(distances)

# helper function to initialise buffer, palette & result vairables in the correct working space
def init_buffers(image: np.ndarray, palette: np.ndarray, colour_space: str) -> tuple:
    height, width, _ = image.shape
    
    # first we need to convert image to our working colour space
    pixels = image.reshape(-1, 3)
    pixels_ws = np.array([cs.to_working_space(p, colour_space) for p in pixels])
    buffer = pixels_ws.reshape(height, width, 3).copy()
    
    # convert palette to working colour space
    palette_ws = np.array([cs.to_working_space(c, colour_space) for c in palette])
    
    # result should always start as zeros in RGB
    result = np.zeros((height, width, 3), dtype=np.float64)
    
    return buffer, palette_ws, result, height, width