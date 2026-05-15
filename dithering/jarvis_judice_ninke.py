import numpy as np
from PIL import Image
from . import colour_space_conversions as cs
from .dithering_utils import to_grayscale, find_nearest_colour, init_buffers

# === Starting with grayscale version & helper function ===

# helper function to diffuse the error to the 4 unprocessed neighbours using the FS kernel
def diffuse_jjn(buffer: np.ndarray, error: np.ndarray, y: int, x: int, height: int, width: int) -> None:
    
    # JJN kernel is bigger than FS, so we need additional checks to account for more neighbours and also be careful with edge pixels
    if x + 1 < width: 
        buffer[y, x + 1] += error * (7 / 48)
    if x + 2 < width:
        buffer[y, x + 2] += error * (5 / 48)
        
    # next row
    if y + 1 < height:
        if x - 2 >= 0:
            buffer[y + 1, x - 2] += error * (3 / 48)
        if x - 1 >= 0:
            buffer[y + 1, x - 1] += error * (5 / 48)
        buffer[y + 1, x] += error * (7 / 48)
        if x + 1 < width:
            buffer[y + 1, x + 1] += error * (5 / 48)
        if x + 2 < width:
            buffer[y + 1, x + 2] += error * (3 / 48)
    
    # next next row
    if y + 2 < height:
        if x - 2 >= 0:
            buffer[y + 2, x - 2] += error * (1 / 48)
        if x - 1 >= 0:
            buffer[y + 2, x - 1] += error * (3 / 48)
        buffer[y + 2, x] += error * (5 / 48)
        if x + 1 < width:
            buffer[y + 2, x + 1] += error * (3 / 48)
        if x + 2 < width:
            buffer[y + 2, x + 2] += error * (1 / 48)

# function to perform the Jarvis-Judice-Ninke dithering algorithm on a grayscale image
def jarvis_judice_ninke_grayscale(gray: np.ndarray) -> np.ndarray:
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
            
            # finally we spread the error to the rest of the unprocessed neighbours using the JJN kernel
            diffuse_jjn(buffer, error, y, x, height, width)
                    
    return result

# === Now for the coloured version ===
            
# function for Floyd-Steinberg dithering on a coloured image with nearest palette colours
def jarvis_judice_ninke_nearest(image: np.ndarray, palette: np.ndarray, colour_space: str = 'rgb') -> np.ndarray:
    
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
            diffuse_jjn(buffer, error, y, x, height, width)
    
    return result

# function for Floyd-Steinberg dithering on a coloured image with RGBXY mixing weights
def jarvis_judice_ninke_weight_driven(image: np.ndarray, palette: np.ndarray, weights: np.ndarray, colour_space: str = 'rgb') -> np.ndarray:
    
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
            diffuse_jjn(buffer, error, y, x, height, width)
    
    return result
    
# function for Floyd-Steinberg dithering on a coloured image with RGBXY mixing weights BUT combining buffer distance and RGBXY weights via alpha
def jarvis_judice_ninke_weighted_nearest(image: np.ndarray, palette: np.ndarray, weights: np.ndarray, colour_space: str = 'rgb', alpha: float = 0.5,) -> np.ndarray:
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
            diffuse_jjn(buffer, error, y, x, height, width)

    return result