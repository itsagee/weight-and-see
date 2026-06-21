import numpy as np

from skimage.metrics import structural_similarity as ssim
from metrics.metrics_utils import to_float

# SSIM (Structural Similarity Index), perceptual metric that quantifies image quality degradation caused by processing such as data compression or by losses in data transmission. It considers changes in structural information, luminance, and contrast between original-reconstructed
# Formula: SSIM(x, y) = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2) / ((mu_x^2 + mu_y^2 + C1) * (sigma_x^2 + sigma_y^2 + C2))
# where mu_x and mu_y are the average of x and y, sigma_x^2 and sigma_y^2 are the variance of x and y, sigma_xy is the covariance of x and y, C1 and C2 are constants to stabilize the division with weak denominator

# for now using the one from the skimage library
def compute_ssim(original: np.ndarray, reconstruction: np.ndarray) -> float:
    
    original = to_float(original)
    reconstruction = to_float(reconstruction)
        
    # channel_axis = 2 to compute SSIM per channel then average
    return ssim(original, reconstruction, data_range=1.0, channel_axis=2)