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
def rgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    # skimage expects shape (1, 1, 3) for a single pixel
    lab = color.rgb2lab(rgb.reshape(1, 1, 3)).reshape(3)
    return lab

# this function is used to convert colours from the CIELAB to the RGB colour space
def lab_to_rgb(lab: np.ndarray) -> np.ndarray:
    rgb = color.lab2rgb(lab.reshape(1, 1, 3)).reshape(3)
    return rgb

# this function is used to convert colours from RGB to CIExyY
def rgb_to_xyY(rgb: np.ndarray) -> np.ndarray: 
    # RGB → XYZ
    xyz = color.rgb2xyz(rgb.reshape(1, 1, 3)).reshape(3)

    # XYZ → xyY
    X, Y, Z = xyz
    denom = X + Y + Z + 1e-8  # avoid division by zero
    x = X / denom
    y = Y / denom
    
    # Y stays as Y (luminance)
    xyY = np.array([x, y, Y])
    
    return xyY

# this function is used to convert colours from CIExyY to RGB
def xyY_to_rgb(xyY: np.ndarray) -> np.ndarray:
    # xyY → XYZ
    x, y, Y = xyY
    X = (Y / (y + 1e-8)) * x
    Z = (Y / (y + 1e-8)) * (1 - x - y)
    xyz = np.array([X, Y, Z])

    # XYZ → RGB
    rgb = color.xyz2rgb(xyz.reshape(1, 1, 3)).reshape(3)
    
    return rgb

# this function is used to convert colours from the RGB to the working colour space (either RGB or CIELAB) depending on the user's choice
def to_working_space(rgb: np.ndarray, colour_space: str) -> np.ndarray:
    if colour_space == 'rgb':
        return rgb
    elif colour_space == 'cielab':
        return rgb_to_lab(rgb)
    elif colour_space == 'ciexyy':
        return rgb_to_xyY(rgb)
    else:
        raise ValueError(f"Unknown colour space: {colour_space}")

# this function is used to convert colours from the working colour space (either RGB or CIELAB) to RGB depending on the user's choice
def to_rgb(pixel: np.ndarray, colour_space: str) -> np.ndarray:
    if colour_space == 'rgb':
        return pixel
    elif colour_space == 'cielab':
        return lab_to_rgb(pixel)
    elif colour_space == 'ciexyy':
        return xyY_to_rgb(pixel)
    else:
        raise ValueError(f"Unknown colour space: {colour_space}")