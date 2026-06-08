import numpy as np
from PIL import Image
from . import colour_space_conversions as cs
from .dithering_utils import to_grayscale, find_nearest_colour, init_buffers

# === Starting with grayscale version & helper function ===

# helper function to diffuse the error to the 4 unprocessed neighbours using the FS kernel
def diffuse_fs(buffer: np.ndarray, error: np.ndarray, y: int, x: int, height: int, width: int) -> None:
    if x + 1 < width:
        buffer[y, x + 1] += error * (7 / 16)
    if y + 1 < height:
        buffer[y + 1, x] += error * (5 / 16)
        if x - 1 >= 0:
            buffer[y + 1, x - 1] += error * (3 / 16)
        if x + 1 < width:
            buffer[y + 1, x + 1] += error * (1 / 16)
            
# helper softmax function
def softmax(x):
    e = np.exp(x - x.max())
    return e / e.sum()
            
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
            diffuse_fs(buffer, error, y, x, height, width)
    return result


# === Now for the coloured version ===
            
# function for Floyd-Steinberg dithering on a coloured image with nearest palette colours
# HERE we are picking palette colour closest in colour space (don't care about weights)
def floyd_steinberg_nearest(image: np.ndarray, palette: np.ndarray, colour_space: str = 'rgb'):
    
    # first we need to convert the image & palete to our workign colour space + initialise all the needed buffers
    buffer, palette_ws, result, height, width = init_buffers(image, palette, colour_space)
    
    # now we can safely move on to processing the image pixel by pixel
    for y in range(height):
        for x in range(width):
            # we first need to find the nearest palette colour for current pixel
            nearest_idx = find_nearest_colour(buffer[y, x], palette_ws, colour_space)
            chosen_colour_ws = palette_ws[nearest_idx]
                        
            # then compute the error and diffuse it
            error = buffer[y, x] - chosen_colour_ws
            diffuse_fs(buffer, error, y, x, height, width)
            
            # finally convert back to RGB and store result 
            result[y, x] = cs.to_rgb(chosen_colour_ws, colour_space)
            
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
                        
            # then compute the error between buffer (not original) and chosen colour
            error = buffer[y, x] - chosen_colour_ws
            diffuse_fs(buffer, error, y, x, height, width)
            
            # finally convert back to RGB and store result 
            result[y, x] = cs.to_rgb(chosen_colour_ws, colour_space)
    
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

            # finally we compute the error and diffuse it as before
            error = buffer[y, x] - chosen_colour_ws
            diffuse_fs(buffer, error, y, x, height, width)
            
            # finally convert back to RGB and store result 
            result[y, x] = cs.to_rgb(chosen_colour_ws, colour_space)

    return result

# function for Floyd-Steinberg dithering on a coloured image with RGBXY mixing weights BUT combining buffer distance and RGBXY weights via softmax
# HERE both distance to palette colours AND RGBXY mixing are converted to probability distributions witht he softmax, then combined using alpha before picking the best colour
def floyd_steinberg_softmax(image: np.ndarray, palette: np.ndarray, weights: np.ndarray, colour_space: str = 'rgb', alpha: float = 0.5) -> np.ndarray:
    """
    As a small note on the alpha: it controls the balance between distance-based and weight-based decisions as follows
    alpha = 0.0 for pure distance-driven (similar to nearest-colour)
    alpha = 1.0 for pure weight-driven (similar to floyd_steinberg_weight_driven)
    alpha = 0.5 for balanced combination (set as our default option)
    
    Note: unlike weighted-nearest, instead of linear norm, we use softmax to convert both to probability distributions before combining,
    so technically combination should be more principled and less sensitive to outliers
    """
    
    # first we need to convert the image & palete to our workign colour space + initialise all the needed buffers
    buffer, palette_ws, result, height, width = init_buffers(image, palette, colour_space)
    
    # as usual pixel by pixel
    for y in range(height):
        for x in range(width):
            
            # we compute the perceptual distances to all palette colours in the working colour space
            distances = cs.compute_distance(buffer[y, x], palette_ws, colour_space)
            
            # high weight = high score, lower distance = high score (negate distances)
            # convert both signals to probability distributions using the softmax function
            weight_score = softmax(weights[y, x])
            distance_score = softmax(-distances)
            
            # combined score: our goal is to maximise this in order to find the best palette colour
            # alpha controls the balance between distance-based and weight-based decisions as follows
            combined = (1 - alpha) * distance_score + alpha * weight_score
            chosen_idx = np.argmax(combined)
            chosen_colour_ws = palette_ws[chosen_idx]
            
            # we compute the error and diffuse it as before
            error = buffer[y, x] - chosen_colour_ws
            diffuse_fs(buffer, error, y, x, height, width)
            
            # finally convert back to RGB and store result 
            result[y, x] = cs.to_rgb(chosen_colour_ws, colour_space)

    return result

# function for Floyd-Steinberg dithering on a coloured image with RGBXY mixing weights influencing error diffusion rather than colour assignment
# HERE we are picking palette colour based on nearest neighbour, BUT the amount of error diffused to neighbours is controlled by the mixing weight of the chosen colour
def floyd_steinberg_error_scaled(image: np.ndarray, palette: np.ndarray, weights: np.ndarray, colour_space: str = 'rgb', mode: str = 'scale', alpha: float = 0.5, p: float = 1.0) -> np.ndarray:
    """
    As a small note,  the weights influence how much quantisation error is propagated to neighbours based on the mode
    
    'scale': we multiply the error by (1-weight), so that high weight = confident assignment = less error diffused
    'weighted_target': we choose the nearest colour to a weighted average of the buffer pixel and the RGBXY mixed colour, so that the error is still fully diffused but the choice of colour is influenced by weights, essentially without losing information
    'confidence': we multiply error by (1-weight^p), where p>1 sharpens and p<1 softens the scalign effect. p=1 would be the equivalent of the 'scale' mode
    """
    
    # first we need to convert the image & palete to our workign colour space + initialise all the needed buffers
    buffer, palette_ws, result, height, width = init_buffers(image, palette, colour_space)
    
    # as usual pixel by pixel
    for y in range(height):
        for x in range(width):
            
            if mode == 'weighted_target':
                # Alternative 3: finding the nearest colour to weighted average of buffer and mixing colour
                # blend buffer value with RGBXY mixed colour as the assignment target
                # full error is still diffused — no information lost
                mixed_colour = np.sum(weights[y, x][:, None] * palette_ws, axis=0)
                target = (1 - alpha) * buffer[y, x] + alpha * mixed_colour
                chosen_idx = find_nearest_colour(target, palette_ws, colour_space)
            else:
                # all other modes use standard nearest-colour assignment
                # we do the standard nearest-colour assignment in the working colour space
                chosen_idx = find_nearest_colour(buffer[y, x], palette_ws, colour_space)
            
            chosen_colour_ws = palette_ws[chosen_idx]
            
            # now on to compute the base error
            base_error = buffer[y, x] - chosen_colour_ws
            
            if mode == 'scale':
                # scale error by (1 - weight): high weight = pixel strongly belongs to this colour (small error diffused), low weight = more of an uncertain assignment (more error diffused)
                # Alternative 1: this one scales the error down --> also losing info
                error = base_error * (1 - weights[y, x, chosen_idx])
            
            elif mode == 'weighted_target':
                # Alternative 2: redistributing by weight
                # instead of diffusing all the error in one direction, we split depending on how much each colour contributes
                error = base_error 
                
            elif mode == 'confidence':
                # Alternative 4: error scaling by confidence interval
                # by raising the weight to a power p to make the scaling more aggressive for high-confidence pixels
                # where p>sharpens (only very high-weight pixels diffuse little error), p<1softens scaling (even moderate-weight pixels diffuse little error)
                confidence = weights[y, x, chosen_idx] ** p
                error = base_error * (1 - confidence)
                
            else:
                raise ValueError(f"Invalid mode: {mode}. Choose from 'scale', 'weighted_target', or 'confidence'.")
            
            # diffuse and store (same in both functions)
            diffuse_fs(buffer, error, y, x, height, width)
            result[y, x] = cs.to_rgb(chosen_colour_ws, colour_space)
            
    return result