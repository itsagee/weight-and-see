"""
dithering_runner.py
~~~~~~~~~~~~~~~~~~~
This file is the Single entry-point for executing all assignment strategies of any dithering algorithm and save the outputs to disk.

Expected module interface
-------------------------
Each algorithm module (floyd_steinberg, jarvis_judice_ninke, stucki, …) must expose the following functions (replace ``<algo>`` with the module-level
prefix, e.g. ``floyd_steinberg``, ``jarvis_judice_ninke``, ``stucki``):

    <algo>_nearest(image, palette, *, colour_space)
    <algo>_weight_driven(image, palette, weights, *, colour_space)
    <algo>_weighted_nearest(image, palette, weights, *, alpha, colour_space)
    <algo>_softmax(image, palette, weights, *, alpha, colour_space)
    <algo>_error_scaled(image, palette, weights, *, colour_space, mode, alpha, p)

The function-name prefix is discovered automatically from the module name, so that no extra configuration is needed.

Typical notebook usage
----------------------
    import importlib
    import dithering.floyd_steinberg as fs
    importlib.reload(fs)

    from dithering.runner import run_all_strategies
    from dithering.io import save_image

    results_fs = run_all_strategies(
        algo_module   = fs,
        image         = image,
        palette       = palette,
        weights       = weights,
        colour_spaces = ['rgb', 'cielab', 'ciexyy'],
        alpha         = 0.5,
        modes         = ['scale', 'weighted_target', 'confidence'],
        p             = 2.0,
        output_dir    = OUTPUT_FS,
        image_name    = IMAGE_NAME,
        prefix        = 'fs',
    )

Return value
------------
A dict with keys: 'nearest', 'weights', 'combined', 'softmax', 'error_scaled'.

    results['nearest']            -> {'rgb': arr, 'cielab': arr, 'ciexyy': arr}
    results['error_scaled']       -> {'scale': {'rgb': arr, ...}, ...}
"""

import os
from types import ModuleType

from dithering.io import save_image
from runners.dithering_viz import plot_colourspace_comparison_per_strategy, plot_all

# this function looks up a function in the given module by constructing its name from the module name and a suffix
def _fn(module: ModuleType, suffix: str):
    """Look up ``<module_name>_<suffix>`` on *module* and return it."""
    name = f"{module.__name__.split('.')[-1]}_{suffix}"
    fn = getattr(module, name, None)
    if fn is None:
        raise AttributeError(
            f"Module '{module.__name__}' has no function '{name}'. "
            "Make sure the function follows the naming convention."
        )
    return fn

def run_all_strategies(
    algo_module: ModuleType,
    image,
    palette,
    weights,
    colour_spaces: list[str],
    alpha: float,
    modes: list[str],
    p: float,
    output_dir: str,
    image_name: str,
    prefix: str,
) -> dict:
    """
    This function runs every assignment strategy for *algo_module* across all colour spaces, save each result to *output_dir*, and return a unified results dict.
    The strategy logic that follows is taken from the original notebook, but is now fully automated and can be run in a single call.

    Parameters
    ----------
    algo_module   : the imported algorithm module (e.g. ``floyd_steinberg``)
    image         : H×W×3 float32 array in [0, 1]
    palette       : N×3 float32 array
    weights       : H×W×N float32 array
    colour_spaces : list of colour-space strings, e.g. ['rgb', 'cielab', 'ciexyy']
    alpha         : blending coefficient used by weighted-nearest and softmax
    modes         : error-scaled mode names, e.g. ['scale', 'weighted_target', 'confidence']
    p             : power parameter for error-scaled confidence mode
    output_dir    : directory to write PNG files into (created if absent)
    image_name    : base name used in output filenames (e.g. 'pigs')
    prefix        : short algorithm tag used in filenames (e.g. 'fs', 'jjn', 'st')

    Returns
    -------
    dict with keys 'nearest', 'weights', 'combined', 'softmax', 'error_scaled'
    """
    os.makedirs(output_dir, exist_ok=True)

    results: dict = {}

    # ------------------------------------------------------------------
    # Nearest-colour
    # ------------------------------------------------------------------
    print(f"[{prefix.upper()}] Running nearest-colour …")
    nearest_fn = _fn(algo_module, "nearest")
    results["nearest"] = {
        cs: nearest_fn(image, palette, colour_space=cs)
        for cs in colour_spaces
    }
    for cs in colour_spaces:
        save_image(
            results["nearest"][cs],
            os.path.join(output_dir, f"{image_name}_{prefix}_nearest_{cs}.png"),
        )

    # ------------------------------------------------------------------
    # Weight-driven
    # ------------------------------------------------------------------
    print(f"[{prefix.upper()}] Running weight-driven …")
    weights_fn = _fn(algo_module, "weight_driven")
    results["weights"] = {
        cs: weights_fn(image, palette, weights, colour_space=cs)
        for cs in colour_spaces
    }
    for cs in colour_spaces:
        save_image(
            results["weights"][cs],
            os.path.join(output_dir, f"{image_name}_{prefix}_weights_{cs}.png"),
        )

    # ------------------------------------------------------------------
    # Weighted-nearest  (alpha-dependent)
    # ------------------------------------------------------------------
    print(f"[{prefix.upper()}] Running weighted-nearest (α={alpha}) …")
    wn_fn = _fn(algo_module, "weighted_nearest")
    results["combined"] = {
        cs: wn_fn(image, palette, weights, alpha=alpha, colour_space=cs)
        for cs in colour_spaces
    }
    for cs in colour_spaces:
        save_image(
            results["combined"][cs],
            os.path.join(output_dir, f"{image_name}_{prefix}_combined_α={alpha}_{cs}.png"),
        )

    # ------------------------------------------------------------------
    # Softmax  (alpha-dependent)
    # ------------------------------------------------------------------
    print(f"[{prefix.upper()}] Running softmax (α={alpha}) …")
    softmax_fn = _fn(algo_module, "softmax")
    results["softmax"] = {
        cs: softmax_fn(image, palette, weights, alpha=alpha, colour_space=cs)
        for cs in colour_spaces
    }
    for cs in colour_spaces:
        save_image(
            results["softmax"][cs],
            os.path.join(output_dir, f"{image_name}_{prefix}_softmax_α={alpha}_{cs}.png"),
        )

    # ------------------------------------------------------------------
    # Error-scaled  (one sub-dict per mode)
    # ------------------------------------------------------------------
    es_fn = _fn(algo_module, "error_scaled")
    results["error_scaled"] = {}
    for mode in modes:
        print(f"[{prefix.upper()}] Running error-scaled ({mode}) …")
        results["error_scaled"][mode] = {
            cs: es_fn(image, palette, weights, colour_space=cs, mode=mode, alpha=alpha, p=p)
            for cs in colour_spaces
        }
        for cs in colour_spaces:
            save_image(
                results["error_scaled"][mode][cs],
                os.path.join(
                    output_dir,
                    f"{image_name}_{prefix}_error_scaled_{mode}_{cs}.png",
                ),
            )

    print(f"[{prefix.upper()}] Done. Results saved to: {output_dir}")
    
    return results
