from dreamsim import dreamsim
from PIL import Image

import torch
import numpy as np

from metrics.metrics_utils import to_float

# --- setups & helper functions ---

model, preprocess = dreamsim(pretrained=True, device='cpu')

# from (H, W, 3) to (1, 3, H, W) in [0, 1]
def to_dreamsim_tensor(image: np.ndarray) -> torch.Tensor:
    pil = Image.fromarray((to_float(image) * 255).astype(np.uint8))
    
    return preprocess(pil).to('cpu')

# --- main function ---

def compute_dreamsim(original: np.ndarray, reconstruction: np.ndarray) -> float:
    
    # DreamSim expects tensors in [0, 1], of shape (1, 3, H, W), so we need to convert our numpy arrays accordingly & normalize
    original_tensor = to_dreamsim_tensor(original)
    reconstruction_tensor = to_dreamsim_tensor(reconstruction)

    with torch.no_grad():
        dreamsim_value = model(original_tensor, reconstruction_tensor)
    
    return dreamsim_value.item()