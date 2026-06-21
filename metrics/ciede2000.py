import numpy as np
import colour
from skimage import color

from metrics.metrics_utils import to_lab

# CIEDE2000 (Color Difference), a color difference formula that quantifies the perceived differences between two colors. It is an improvement over previous color difference formulas and is designed to be more perceptually uniform, meaning that the same numerical difference corresponds to a similar perceived difference in color.
# Formula: CIEDE2000 = sqrt((delta_L / (k_L * S_L))^2 + (delta_C / (k_C * S_C))^2 + (delta_H / (k_H * S_H))^2 + R_T * (delta_C / (k_C * S_C)) * (delta_H / (k_H * S_H)))
# where delta_L, delta_C, and delta_H are the differences in lightness, chroma, and hue between the two colors, S_L, S_C, and S_H are the weighting functions for lightness, chroma, and hue, k_L, k_C, and k_H are the parametric weighting factors (usually set to 1), and R_T is a rotation term that accounts for the interaction between chroma and hue differences.

# for now using the one from the colour-science library
def compute_ciede2000(original: np.ndarray, reconstruction: np.ndarray) -> float:
    
    # instead of using my own rgb_to_lab, here we have the same thing but faster
    # same underlying conversion as rgb_to_lab, vectorized over the whole image
    # includes the check to ensure the input is in [0, 1] and float64
    orig_lab_flat  = to_lab(original).reshape(-1, 3)
    recon_lab_flat = to_lab(reconstruction).reshape(-1, 3)

    # delta_E returns a per-pixel array, we take the mean
    delta_e = colour.delta_E(orig_lab_flat, recon_lab_flat, method='CIE 2000')
    
    return float(np.mean(delta_e))