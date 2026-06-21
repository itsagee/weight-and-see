import numpy as np

from metrics.metrics_utils import to_float

# PSNR (Peak Signal-to-Noise Ratio), normally used to evaluate the quality of reconstructed images compared to the original images
# Formula: PSNR = 10 * log10(MAX_I ** 2 / MSE) = 20 * log10(MAX_I / sqrt(MSE))
# where MAX_I is the maximum possible pixel value of the image (IF normalized images typically 1.0), and MSE is the Mean Squared Error between original-reconstructed

# for now implementing my own (also exists on the skimage library)
def compute_psnr(original: np.ndarray, reconstruction: np.ndarray) -> float:
    
    # adding this to ensure that the input images are in the range [0, 1], if not we can scale them accordingly
    original = to_float(original)
    reconstruction = to_float(reconstruction)
        
    mse = np.mean((original - reconstruction) ** 2)
    
    if mse == 0:
        # No error, PSNR is infinite, means we have identical images
        return float('inf') 
    return 20 * np.log10(1.0 / np.sqrt(mse))