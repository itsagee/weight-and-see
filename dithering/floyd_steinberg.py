import numpy as np
from PIL import Image
from . import colour_space_conversions as cs

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

# helper function to diffuse the error to the 4 unprocessed neighbours using the FS kernel
def diffuse_error(buffer: np.ndarray, error: np.ndarray, y: int, x: int, height: int, width: int) -> None:
    if x + 1 < width:
        buffer[y, x + 1] += error * (7 / 16)
    if y + 1 < height:
        buffer[y + 1, x] += error * (5 / 16)
        if x - 1 >= 0:
            buffer[y + 1, x - 1] += error * (3 / 16)
        if x + 1 < width:
            buffer[y + 1, x + 1] += error * (1 / 16)
            
# function for Floyd-Steinberg dithering on a coloured image with nearest palette colours
# HERE we are picking palette colour closest in colour space (don't care about weights)
def floyd_steinberg_nearest(image: np.ndarray, palette: np.ndarray, colour_space: str = 'rgb'):
    
    # first we need to convert the image & palete to our workign colour space + initialise all the needed buffers
    buffer, palette_ws, result, height, width = init_buffers(image, palette, colour_space)
    
    # now we can safely move on to processing the image pixel by pixel
    for y in range(height):
        for x in range(width):
            # we first need to find the nearest palette colour for current pixel
            nearest_idx = find_nearest_colour(buffer[y, x], palette_ws)
            chosen_colour_ws = palette_ws[nearest_idx]
            
            # convert back to RGB and store result
            result[y, x] = cs.to_rgb(chosen_colour_ws, colour_space)
            
            # then compute the error and diffuse it
            error = buffer[y, x] - chosen_colour_ws
            diffuse_error(buffer, error, y, x, height, width)
    
    return result

# function for Floyd-Steinberg dithering on a coloured image with RGBXY mixing weights
# HERE we are picking palette colour with the highest mixing weight at this pixel (don't care about distance in colour space)
def floyd_steinberg_weight_driven(image: np.ndarray, palette: np.ndarray, weights: np.ndarray, colour_space: str = 'rgb') -> np.ndarray:

    # first we need to convert the image & palete to our workign colour space + initialise all the needed buffers
    buffer, palette_ws, result, height, width = init_buffers(image, palette, colour_space)
    
    # again pixel by pixel
    for y in range(height):
        for x in range(width):
            # this time we compute the mixed colour using the RGBXY weights (from .js file saved from RGBXY)
            # first we pick the palette colour with the highest weight at this pixel
            chosen_idx = np.argmax(weights[y, x])
            chosen_colour_ws = palette_ws[chosen_idx]
            
            # convert back to RGB and store result
            result[y, x] = cs.to_rgb(chosen_colour_ws, colour_space)
            
            # then compute the error between buffer (not original) and chosen colour
            error = buffer[y, x] - chosen_colour_ws
            diffuse_error(buffer, error, y, x, height, width)
    
    return result

# function for Floyd-Steinberg dithering on a coloured image with RGBXY mixing weights BUT combining buffer distance and RGBXY weights via alpha
# HERE we are picking palette colour with the best combined score of distance in colour space and RGBXY weight at this pixel
def floyd_steinberg_weighted_nearest(image: np.ndarray, palette: np.ndarray, weights: np.ndarray, colour_space: str = 'rgb', alpha: float = 0.5,) -> np.ndarray:
    """
    As a small note on the alpha: it controls the balance between distance-based and weight-based decisions as follows
    alpha = 0.0 for pure nearest-colour (same as floyd_steinberg_nearest)
    alpha = 1.0 for pure weight-driven  (same as floyd_steinberg_weight_driven)
    alpha = 0.5 for balanced combination (set as our default option)
    """
       
    # first we need to convert the image & palete to our workign colour space + initialise all the needed buffers
    buffer, palette_ws, result, height, width = init_buffers(image, palette, colour_space)

    # as usual pixel by pixel
    for y in range(height):
        for x in range(width):
            # we compute the distances to all palette colours from the current buffer pixel
            distances = np.linalg.norm(palette_ws - buffer[y, x], axis=1)

            # we then normalise both signals to [0, 1] so they're comparable
            distances_norm = distances / (distances.max() + 1e-8)
            weights_norm   = weights[y, x] / (weights[y, x].max() + 1e-8)

            # for the combined score, we add the low distance with high weight to get the best candidate
            scores = (1 - alpha) * distances_norm + alpha * (1 - weights_norm)
            chosen_idx = np.argmin(scores)
            chosen_colour_ws = palette_ws[chosen_idx]
            
            # convert back to RGB and store result
            result[y, x] = cs.to_rgb(chosen_colour_ws, colour_space)

            # finally we compute the error and diffuse it as before
            error = buffer[y, x] - chosen_colour_ws
            diffuse_error(buffer, error, y, x, height, width)

    return result