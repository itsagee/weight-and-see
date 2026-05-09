"""
This file is used to load the data from the RGBXY palette decomposition that will be used for dithering.

Expected inputs from the RGBXY codebase that will be used here:
    - <image>-automatic computed palette-modified.js  -> includes the palette colours of the image
    - <image>-weights.js                              -> has the per-pixel mixing weights
    - <image>.jpg (or .png)                           -> is our original image
"""

import json
import numpy as np
from PIL import Image

# this function loads the colours from the RGBXY palette .js file
def load_palette(palette_path: str) -> np.ndarray:
    # first we load the palette from the JSON file, from the 'vs' key that contains said colours
    with open(palette_path, 'r') as f:
        data = json.load(f)

    # we save it as a numpy array of shape (num_colours, 3), with values in [0, 1], in the RGB colour space
    palette = np.array(data['vs'], dtype=np.float64) / 255.0
    
    # sanity check: the palette should have 256 colours, we print the actual number just in case
    print(f"Loaded palette with {len(palette)} colours.")
    return palette

# this function loads the per-pixel mixing weights from the RGBXY weights .js file
def load_weights(weights_path: str) -> np.ndarray:
    # first we load the weights from the JSON file, from the 'weights' key that contains the per-pixel mixing weights
    with open(weights_path, 'r') as f:
        data = json.load(f)

    # we save it as a numpy array of shape (height, width, num_colours), with values in [0, 1], where num_colours is the same as the number of colours in the palette
    weights = np.array(data['weights'], dtype=np.float64)
    
    # sanity check: we print the shape of the weights to confirm it matches our expectations (height, width, num_colours)
    print(f"Loaded weights with shape: {weights.shape} "
          f"(height={weights.shape[0]}, width={weights.shape[1]}, "
          f"num_colours={weights.shape[2]})")
    return weights

# this function loads an image as a numpy array
def load_image(image_path: str) -> np.ndarray:
    # first we load the image using the PIL library, while also converting it to RGB ( in case it's RGBA or grayscale)
    img = Image.open(image_path).convert('RGB')
    # we can now convert it to a numpy array of shape (height, width, 3) with values in [0, 1]
    image = np.array(img, dtype=np.float64) / 255.0
    
    # sanity check: we print the shape of the image to confirm it matches our expectations (height, width, 3)
    print(f"Loaded image with shape: {image.shape} "
          f"(height={image.shape[0]}, width={image.shape[1]})")
    return image

# this function reconstructs the image from the weights and palette using additive blending (the weighted sum of palette colours) in RGBXY, from the RGBXY paper
def reconstruct_additive(weights: np.ndarray, palette: np.ndarray) -> np.ndarray:
    # for each pixel (h, w), multiply each palette colour (c, d) by its weight (c, ) and add them all together.
    reconstruction = np.einsum('hwc,cd->hwd', weights, palette)
    # this is the big version, the upper line is a short version of the same nested matrix operation
    # reconstruction = np.zeros((height, width, 3))
    # for h in range(height):
    #     for w in range(width):
    #         for c in range(num_colours):
    #             reconstruction[h, w] += weights[h, w, c] * palette[c]
    
    # finally, we clamp values between 0 and 1, in case we have any out-of-bounds values 
    reconstruction = np.clip(reconstruction, 0.0, 1.0)
    return reconstruction

# this function saves a numpy image array as an image to the given directory
def save_image(image: np.ndarray, output_path: str) -> None:
    # we first convert the image from numpy array with values in [0, 1] to a PIL image with values in [0, 255] (back to RGB)
    img = Image.fromarray((image * 255).astype(np.uint8))
    # now we can then we save it to the output path
    img.save(output_path)
    
    # sanity check: we print a confirmation along with the output path
    print(f"Saved image to {output_path}")