import numpy as np
import torch

from skimage import color

# to ensure an image is float64 in [0, 1], regardless of whether it was originally uint8 [0, 255] or already float [0, 1]
def to_float(image: np.ndarray) -> np.ndarray:
    image = image.astype(np.float64)
    
    if image.max() > 1.0:
        image = image / 255.0
        
    return image

# to convert an RGB image to CIELAB colour space, input should be in [0, 1]
def to_lab(image: np.ndarray) -> np.ndarray:    
    return color.rgb2lab(to_float(image))