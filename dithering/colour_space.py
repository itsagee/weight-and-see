from skimage import color
import numpy as np

"""
This file contains functions for converting between colour spaces, as well as some helper functions for working with colours. 
It is used by the main script to load the palette and weights, and to reconstruct the image from the weights and palette.

Note:
There are different libraries implementing such conversions, i.e. PIL, openCV,
I chose skimage because their functions work directly with float64 arrays (what I'm using),
which means we can safely avoid any extra conversion steps and potential loss of precision that could come with
"""

# this function is used to convert colours from the RGB to the CIELAB colour space
def rgb_to_cielab(rgb: np.ndarray) -> np.ndarray:
    # skimage expects shape (1, 1, 3) for a single pixel
    cielab = color.rgb2lab(rgb.reshape(1, 1, 3)).reshape(3)
    return cielab

# this function is used to convert colours from the CIELAB to the RGB colour space
def cielab_to_rgb(lab: np.ndarray) -> np.ndarray:
    rgb = color.lab2rgb(lab.reshape(1, 1, 3)).reshape(3)
    return rgb
