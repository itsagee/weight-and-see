import numpy as np
from PIL import Image

# === Starting with grayscale version & helper function ===

# helper function to turn an image into grayscale using standard luminance weights
def to_grayscale(image: np.ndarray) -> np.ndarray:
    return 0.2126 * image[:, :, 0] + 0.7152 * image[:, :, 1] + 0.0722 * image[:, :, 2]

# function to perform the Floyd-Steinberg dithering algorithm on a grayscale image
def floyd_steinberg_grayscale(gray: np.ndarray) -> np.ndarray:
    height, width = gray.shape
    buffer = gray.astype(np.float64).copy()
    result = np.zeros((height, width), dtype=np.float64)
    
    # for all pixels from top-left to bottom-right
    for y in range(height):
        for x in range(width):
            # we read current value from buffer (NOT original) & round
            old_val = buffer[y, x]
            new_val = 1.0 if old_val >= 0.5 else 0.0
            result[y, x] = new_val
            # error shows our decision
            error = old_val - new_val
            
            # finally we spread the error to the 4 unprocessed neighbours
            if x + 1 < width:
                buffer[y, x + 1] += error * (7 / 16)
            if y + 1 < height:
                buffer[y + 1, x] += error * (5 / 16)
                if x - 1 >= 0:
                    buffer[y + 1, x - 1] += error * (3 / 16)
                if x + 1 < width:
                    buffer[y + 1, x + 1] += error * (1 / 16)
    return result


# === Now for the coloured version ===

# helper function to find the nearest palette colour for a given pixel
def find_nearest_colour(pixel: np.ndarray, palette: np.ndarray) -> int:
    # first we need to compute the distances to all palette colours
    distances = np.linalg.norm(palette - pixel, axis=1)
    # so we can return the index of nearest colour
    return np.argmin(distances)

# helper function to diffuse the error to the 4 unprocessed neighbours using the FS kernel
def diffuse_error(buffer: np.ndarray, error: float, y: int, x: int, height: int, width: int):
    if x + 1 < width:
        buffer[y, x + 1] += error * (7 / 16)
    if y + 1 < height:
        buffer[y + 1, x] += error * (5 / 16)
        if x - 1 >= 0:
            buffer[y + 1, x - 1] += error * (3 / 16)
        if x + 1 < width:
            buffer[y + 1, x + 1] += error * (1 / 16)
            
# function for Floyd-Steinberg dithering on a coloured image with nearest palette colours
def floyd_steinberg_nearest(image: np.ndarray, palette: np.ndarray) -> np.ndarray:
    height, width, _ = image.shape
    buffer = image.astype(np.float64).copy()
    result = np.zeros((height, width, 3), dtype=np.float64)
    
    # now to process the image pixel by pixel
    for y in range(height):
        for x in range(width):
            # we first need to find the nearest palette colour for current pixel
            nearest_idx = find_nearest_colour(buffer[y, x], palette)
            nearest_colour = palette[nearest_idx]
            result[y, x] = nearest_colour
            
            # then compute the error and diffuse it
            error = buffer[y, x] - nearest_colour
            diffuse_error(buffer, error, y, x, height, width)
    
    return result

# function for Floy-Steinberg dithering on a coloured image with RGBXY mixing weights
def floyd_steinberg_weight_driven(image: np.ndarray, palette: np.ndarray, weights: np.ndarray) -> np.ndarray:
    height, width, _ = image.shape
    buffer = image.astype(np.float64).copy()
    result = np.zeros((height, width, 3), dtype=np.float64)
    
    # again pixel by pixel
    for y in range(height):
        for x in range(width):
            # this time we compute the mixed colour using the RGBXY weights (from .js file saved from RGBXY)
            # first we pick the palette colour with the highest weight at this pixel
            chosen_idx = np.argmax(weights[y, x])
            chosen_colour = palette[chosen_idx]
            result[y, x] = chosen_colour

            # then compute the error between buffer (not original) and chosen colour
            error = buffer[y, x] - chosen_colour
            diffuse_error(buffer, error, y, x, height, width)
    
    return result

# function for Floyd-Steinberg dithering on a coloured image with RGBXY mixing weights BUT combining buffer distance and RGBXY weights via alpha
def floyd_steinberg_weighted_nearest(image: np.ndarray, palette: np.ndarray, weights: np.ndarray, alpha: float = 0.5,) -> np.ndarray:
    """
    As a small note on the alpha: it controls the balance between distance-based and weight-based decisions as follows
    alpha = 0.0 for pure nearest-colour (same as floyd_steinberg_nearest)
    alpha = 1.0 for pure weight-driven  (same as floyd_steinberg_weight_driven)
    alpha = 0.5 for balanced combination (set as our default option)
    """
    
    height, width, _ = image.shape
    buffer = image.astype(np.float64).copy()
    result = np.zeros((height, width, 3), dtype=np.float64)

    # as usual pixel by pixel
    for y in range(height):
        for x in range(width):
            # we compute the distances to all palette colours from the current buffer pixel
            distances = np.linalg.norm(palette - buffer[y, x], axis=1)

            # we then normalise both signals to [0, 1] so they're comparable
            distances_norm = distances / (distances.max() + 1e-8)
            weights_norm   = weights[y, x] / (weights[y, x].max() + 1e-8)

            # for the combined score, we add the low distance with high weight to get the best candidate
            scores = (1 - alpha) * distances_norm + alpha * (1 - weights_norm)
            chosen_idx = np.argmin(scores)
            chosen_colour = palette[chosen_idx]
            result[y, x] = chosen_colour

            # finally we compute the error and diffuse it as before
            error = buffer[y, x] - chosen_colour
            diffuse_error(buffer, error, y, x, height, width)

    return result