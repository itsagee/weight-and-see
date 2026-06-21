import numpy as np
import colour

from scipy.ndimage import gaussian_filter
from skimage import color
from metrics.metrics_utils import to_float

# --- setting up ---

# transformation matrix from XYZ to opponent channels (from Zhang & Wandell, 1997 & Johnson & Fairchild, 2003 eq. 1)
# we'll compute it only once at module level to avoid recomputing on every call
M = np.array([
    # [ 0.279,  0.720,  -0.107], # achromatic # AS PER 1997 PAPER)
    [ 0.297,  0.720,  -0.107], # achromatic # AS PER 2003 PAPER)
    [ -0.449, 0.290, -0.077],  # red-green
    [ 0.086, -0.590,  0.501],  # blue-yellow
])
# using this from the paper rather than the np.linalg.inv of M (explicit inverse from Johnson & Fairchild, 2003 eq. 7)
M_inv = np.array([
    [ 0.979, -1.535,  0.445],
    [ 1.189,  0.764,  0.135],
    [ 1.232,  1.163,  2.079],
])

# Kernel parameters from Johnson & Fairchild (2003) Table I
# Each tuple: (weight w_i, spread sigma_i in degrees of visual angle)
# Achromatic: 3-Gaussian band-pass (note negative weight on third)
# Chromatic:  2-Gaussian low-pass
KERNEL_PARAMS = {
    'achromatic': [(1.00327, 0.0500), (0.11442, 0.2250), (-0.11769, 7.0000)],
    'red_green': [(0.61673, 0.0685), (0.38328, 0.8260)],
    'blue_yellow': [(0.56789, 0.0920), (0.43212, 0.6451)],
}
     
# --- helper functions ---                                 

# XYZ (H, W, 3) to opponent channels AC1C2 (H, W, 3). based on Eq. 1
def opponent_channels(xyz: np.ndarray) -> np.ndarray:

    h, w, _ = xyz.shape
    flat = xyz.reshape(-1, 3)
    opp = (M @ flat.T).T
    return opp.reshape(h, w, 3)

# opponent channels AC1C2 (H, W, 3) back to XYZ (H, W, 3). based on Eq. 7
def opponent_to_xyz(opp: np.ndarray) -> np.ndarray:

    h, w, _ = opp.shape
    flat = opp.reshape(-1, 3)
    xyz = (M_inv @ flat.T).T
    return xyz.reshape(h, w, 3)

# with this function we can apply the weighted sum of gaussians filter to one opponent channel
# essentially implementing Eqs. 2-3: filter = k * sum_i( w_i * G(sigma_i) )
def apply_csf_filter(channel: np.ndarray, params: list, pixels_per_degree: float) -> np.ndarray:
    
    result = np.zeros_like(channel)
    
    # for each weight and spread in degrees, we're converting spread to pixels and applying gaussian filter, accumulating the weighted result
    for weight, spread_deg in params:
        sigma_px = spread_deg * pixels_per_degree
        # doing 'reflect' mode to avoid boundary artefacts from the wide achromatic Gaussian
        result += weight * gaussian_filter(channel, sigma=sigma_px, mode='reflect')
    return result

# with this we apply per-channel CSF filters to all three opponent channels, essentially calling the above function three times with different parameters
def spatial_filter(opp: np.ndarray, pixels_per_degree: float) -> np.ndarray:

    filtered = np.zeros_like(opp)
    
    # for each channel, we call the apply_csf_filter with the appropriate parameters from KERNEL_PARAMS
    filtered[:, :, 0] = apply_csf_filter(opp[:, :, 0], KERNEL_PARAMS['achromatic'], pixels_per_degree)
    filtered[:, :, 1] = apply_csf_filter(opp[:, :, 1], KERNEL_PARAMS['red_green'], pixels_per_degree)
    filtered[:, :, 2] = apply_csf_filter(opp[:, :, 2], KERNEL_PARAMS['blue_yellow'], pixels_per_degree)
    
    return filtered

# --- Main function ---

# S-CIELAB (Spatial CIELAB), an extension of the CIELAB color difference formula that incorporates spatial information. Designed to account for the fact that human perception of color differences can be influenced by the spatial arrangement of colors in an image.
# Essentially, we apply spatial CSF pre-filtering in opponent color space and then computing the mean CIEDE2000 color difference between the original and reconstructed images.

# this is my own function, not from a library, implementation of the S-CIELAB metric as described in the paper by Zhang & Wandell (1997) and Johnson & Fairchild (2003).
# actual setup using Johnson & Fairchild (2003) Eq. 4: px/deg = ppi / ((180/pi) * arctan(1 / viewing_distance_inches))
def compute_scielab(original: np.ndarray, reconstruction: np.ndarray, pixels_per_degree: float = 60.0) -> float:
    
    original = to_float(original)
    reconstruction = to_float(reconstruction)
    
    # step 1: from RGB TO XYZ
    original_xyz = color.rgb2xyz(original)
    reconstruction_xyz = color.rgb2xyz(reconstruction)
    
    # step 2: xyz to the opponent channels (Eq. 1)
    original_opp = opponent_channels(original_xyz)
    reconstruction_opp = opponent_channels(reconstruction_xyz)
    
    # step 3: spatial filtering per channel (Table I, Eqs. 2-3)
    original_filt = spatial_filter(original_opp, pixels_per_degree)
    reconstruction_filt = spatial_filter(reconstruction_opp, pixels_per_degree)
    
    # step 4: from opponent channels to xyz (Eq. 7) to Lab
    original_lab = color.xyz2lab(opponent_to_xyz(original_filt))
    reconstruction_lab = color.xyz2lab(opponent_to_xyz(reconstruction_filt))
    
    # step 5: compute CIEDE2000 per pixel, return mean (mostly because the paper recommends CIEDE2000 over the DE76, in the "Color Difference Formula" section)
    delta_e = colour.delta_E(original_lab.reshape(-1, 3), reconstruction_lab.reshape(-1, 3), method='CIE 2000')
    
    # # step 5 - ALTERNATIVE
    # # in case we want to use the DE76, uncomment this & comment CIEDE2000 lines
    # delta_e = np.sqrt(np.sum((original_lab - reconstruction_lab) ** 2, axis=2))
    
    return float(np.mean(delta_e))


if __name__ == '__main__':
    # little sanity check: neutral grey should give near-zero opponent response
    grey_xyz = color.rgb2xyz(np.array([[[0.5, 0.5, 0.5]]]))
    opp = opponent_channels(grey_xyz)
    # channels 1 and 2 should be close to 0
    print(opp)