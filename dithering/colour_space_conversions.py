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
    
# should i change to perceptual difference? -->
def compute_distance(pixel: np.ndarray, palette: np.ndarray, colour_space: str) -> np.ndarray:
    """
    Compute distances between a pixel and all palette colours.
    Both pixel and palette are assumed to already be in the given colour_space.
    
    RGB:    Euclidean, fast but not perceptually uniform
    CIELAB: Euclidean = ΔE76, reasonable perceptual metric
    CIExyY: three different alternatives were tested, for now we just do Euclidean in xyY, but this is not perceptually uniform and may not be the best choice. 
            A better approach would be to convert to CIELAB for distance computation only, since CIELAB is designed for perceptual uniformity and has a standardised distance metric (ΔE76).
    """
    
    if (colour_space == 'rgb') or (colour_space == 'cielab'):
        # Euclidean in RGB, simple metric, but not perceptually uniform
        # Euclidean in CIELAB = ΔE76, reasonable perceptual metric
        return np.linalg.norm(palette - pixel, axis=1)

    elif colour_space == 'ciexyy':
        # # ALTERNATIVE 1
        # # weight luminance and chromaticity separately
        # # Y (luminance) is index 2, x and y are indices 0 and 1
        # chroma_diff = np.linalg.norm(palette[:, :2] - pixel[:2], axis=1)
        # luma_diff   = np.abs(palette[:, 2] - pixel[2])
        
        # # luminance contributes more to perceived difference
        # return 2.0 * luma_diff + chroma_diff
        
        # ========================
        
        # # # ALTERNATIVE 2
        # # xyY is not perceptually uniform, no standard distance metric exists for it
        
        # # xyY has no standard perceptual distance metric, Euclidean in xyY is misleading because equal steps in chromaticity (x, y) do not correspond to equal perceived differences (CIE 1931 xy is non-uniform).
        # # let's try converting to CIELAB for distance computation only since CIELAB is designed for perceptual uniformity and has a standardised distance metric (ΔE76)
        # # Inputs are in xyY; the conversion is: xyY → RGB → CIELAB.
        # pixel_rgb   = xyY_to_rgb(pixel)
        # palette_rgb = np.array([xyY_to_rgb(c) for c in palette])

        # pixel_rgb   = np.clip(xyY_to_rgb(pixel), 0, 1)
        # palette_rgb = np.clip(np.array([xyY_to_rgb(c) for c in palette]), 0, 1)

        # pixel_lab   = color.rgb2lab(pixel_rgb.reshape(1, 1, 3)).reshape(3)
        # palette_lab = np.array([
        #     color.rgb2lab(c.reshape(1, 1, 3)).reshape(3)
        #     for c in palette_rgb
        # ])

        # return np.linalg.norm(palette_lab - pixel_lab, axis=1)
        
        # ========================
        
        # ALT 3
        return np.linalg.norm(palette - pixel, axis=1)

    else:
        raise ValueError(f"Unknown colour space: {colour_space}")