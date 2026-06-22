import lpips
import torch
import numpy as np

from metrics.metrics_utils import to_float

# --- setups & helper functions ---

# could also use 'vgg'
loss_fn = lpips.LPIPS(net='alex').to('cpu')

# from (H, W, 3) to (1, 3, H, W) in [-1, 1]
def to_tensor(image: np.ndarray) -> torch.Tensor:
    # scale to [-1, 1]
    image = (to_float(image) * 2.0) - 1.0 
    
    return torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float().cpu()

# --- main function ---

def compute_lpips(original: np.ndarray, reconstruction: np.ndarray) -> float:
    
    # LPIPS expects tensors in [-1, 1], of shape (1, 3, H, W), so we need to convert our numpy arrays accordingly & normalize
    original_tensor = to_tensor(original)
    reconstruction_tensor = to_tensor(reconstruction)

    with torch.no_grad():
        lpips_value = loss_fn(original_tensor, reconstruction_tensor)
    
    return lpips_value.item()