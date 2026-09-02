"""
dithering/viz.py
~~~~~~~~~~~~~~~~
A lot of reusable visualisation helpers for dithering comparisons.

All functions accept a `results` dict with the structure produced by `dithering.runner.run_all_strategies`:

    results = {
        'nearest':      {'rgb': <H×W×3 array>, 'cielab': ..., 'ciexyy': ...},
        'weights':      {...},
        'combined':     {...},
        'softmax':      {...},
        'error_scaled': {
            'scale':           {'rgb': ..., ...},
            'weighted_target': {...},
            'confidence':      {...},
        },
    }

Typical notebook usage
----------------------
    from dithering.viz import plot_all

    plot_all(results_fs, image, colour_spaces, alpha, modes,
             title_prefix='Floyd-Steinberg', zoom=(220, 450, 470, 700))
"""

import os
import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _build_strategies_list(results: dict, alpha: float, modes: list) -> list:
    """This function returns the ordered (label, per-cs-dict) pairs used by several plots."""
    strategies = [
        ("Nearest-colour",              results["nearest"]),
        ("Weight-driven",               results["weights"]),
        (f"Weighted-nearest (α={alpha})", results["combined"]),
        (f"Softmax (α={alpha})",          results["softmax"]),
    ]
    for mode in modes:
        strategies.append((f"Error-scaled ({mode})", results["error_scaled"][mode]))
        
    return strategies


# ---------------------------------------------------------------------------
# Public plotting functions
# ---------------------------------------------------------------------------

def plot_bw_preview(image, gray, dithered_bw, title_prefix=""):
    """
    Show original → greyscale → B&W dithered side-by-side.

    Parameters
    ----------
    image       : H×W×3 float array  – original RGB image
    gray        : H×W float array    – greyscale version
    dithered_bw : dict keyed by algo name, e.g. {'fs': arr, 'jjn': arr, 'st': arr}
                  OR a single H×W array if there is only one algorithm.
    title_prefix : str – used as figure suptitle prefix
    """
    if not isinstance(dithered_bw, dict):
        dithered_bw = {title_prefix: dithered_bw}

    n = len(dithered_bw)
    fig, axes = plt.subplots(1, 2 + n, figsize=(6 * (2 + n), 5))

    axes[0].imshow(image)
    axes[0].set_title("Original")
    axes[0].axis("off")

    axes[1].imshow(gray, cmap="gray")
    axes[1].set_title("Grayscale")
    axes[1].axis("off")

    for ax, (label, arr) in zip(axes[2:], dithered_bw.items()):
        ax.imshow(arr, cmap="gray", interpolation="nearest")
        ax.set_title(f"{label} (black & white)")
        ax.axis("off")

    if title_prefix:
        fig.suptitle(title_prefix, fontsize=13)
    plt.tight_layout()
    plt.show()

def plot_strategy_comparison_per_cs(
    results,
    image,
    colour_spaces,
    alpha,
    modes,
    title_prefix="",
):
    """
    For each colour space we need one row showing Original + all assignment strategies.

    This function produces one figure per colour space.
    """
    
    main_strategies = [
        ("Nearest-colour",                results["nearest"]),
        ("Weight-driven",                 results["weights"]),
        (f"Weighted-nearest (α={alpha})", results["combined"]),
        (f"Softmax (α={alpha})",          results["softmax"]),
    ]

    for cs in colour_spaces:
        fig, axes = plt.subplots(1, len(main_strategies) + 1, figsize=(22, 5))
        fig.suptitle(f"{title_prefix} - colour space: {cs.upper()}", fontsize=13)

        axes[0].imshow(image)
        axes[0].set_title("Original")
        axes[0].axis("off")

        for ax, (label, res_dict) in zip(axes[1:], main_strategies):
            ax.imshow(res_dict[cs], interpolation="nearest")
            ax.set_title(label)
            ax.axis("off")

        plt.tight_layout()
        plt.show()


def plot_error_scaled_per_cs(
    results,
    image,
    colour_spaces,
    modes,
    title_prefix="",
):
    """
    For each colour space we need the original + each error-scaled mode side-by-side.

    This function produces one figure per colour space.
    """
    
    for cs in colour_spaces:
        fig, axes = plt.subplots(1, len(modes) + 1, figsize=(20, 5))
        fig.suptitle(
            f"{title_prefix} Error-scaled modes comparison - {cs.upper()}", fontsize=13
        )

        axes[0].imshow(image)
        axes[0].set_title("Original")
        axes[0].axis("off")

        for ax, mode in zip(axes[1:], modes):
            ax.imshow(results["error_scaled"][mode][cs], interpolation="nearest")
            ax.set_title(f"error-scaled\n({mode})")
            ax.axis("off")

        plt.tight_layout()
        plt.show()


def plot_colourspace_comparison_per_strategy(
    results,
    image,
    colour_spaces,
    alpha,
    modes,
    title_prefix="",
    save_path=None,
    dpi=150,
):
    """
    Finally for the one BIG grid: one rows per strategy, one columns for the original and one for each colour space.
    Each row has a rotated label on the left, and everythign is compiled to one figure total.

    Parameters
    ----------
    save_path : optional file path (e.g. 'output/fs_comparison.png').
                Supports any matplotlib format: .png, .pdf, .svg.
                If None the figure is only shown, not saved.
    dpi       : resolution used when saving (default 150).

    Returns
    -------
    The matplotlib Figure, so callers can save or embed it further.
    """
    
    strategies = _build_strategies_list(results, alpha, modes)
    n_rows = len(strategies)
    n_cols = len(colour_spaces) + 2

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(18, 4 * n_rows),
        gridspec_kw={"width_ratios": [0.1] + [1] * (n_cols - 1)},
    )

    fig.suptitle(f"{title_prefix} - strategy comparison", fontsize=14, fontweight="bold")

    col_titles = ["Original"] + [cs.upper() for cs in colour_spaces]
    for col, title in enumerate(col_titles, start=1):
        axes[0, col].set_title(title, fontsize=11, pad=6)

    for row, (strategy_name, res_dict) in enumerate(strategies):
        # Col 0: rotated row label
        axes[row, 0].set_axis_off()
        axes[row, 0].text(
            0.5, 0.5,
            strategy_name,
            transform=axes[row, 0].transAxes,
            fontsize=10,
            va="center",
            ha="center",
            rotation=90,
            wrap=True,
        )

        # Col 1: original
        axes[row, 1].imshow(image)
        axes[row, 1].axis("off")

        # Remaining cols: one per colour space
        for col, cs in enumerate(colour_spaces, start=2):
            axes[row, col].imshow(res_dict[cs], interpolation="nearest")
            axes[row, col].axis("off")

    plt.tight_layout(rect=[0, 0, 1, 0.98])

    if save_path is not None:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")

    plt.show()
    return fig


def plot_zoomed_comparison(
    results,
    image,
    colour_spaces,
    alpha,
    modes,
    zoom,
    title_prefix="",
):
    """
    Zoomed-region comparison: main strategies then error-scaled modes.

    Parameters
    ----------
    zoom : (x1, y1, x2, y2) pixel coordinates of the region to crop.
    """
    x1, y1, x2, y2 = zoom

    main_strategies = [
        ("Nearest-colour",                results["nearest"]),
        ("Weight-driven",                 results["weights"]),
        (f"Weighted-nearest (α={alpha})", results["combined"]),
        (f"Softmax (α={alpha})",           results["softmax"]),
    ]

    # --- main strategies zoomed ---
    for cs in colour_spaces:
        fig, axes = plt.subplots(1, len(main_strategies) + 1, figsize=(22, 5))
        fig.suptitle(f"{title_prefix} zoomed - {cs.upper()}", fontsize=13)

        axes[0].imshow(image[y1:y2, x1:x2], interpolation="nearest")
        axes[0].set_title("Original (zoomed)")
        axes[0].axis("off")

        for ax, (label, res_dict) in zip(axes[1:], main_strategies):
            ax.imshow(res_dict[cs][y1:y2, x1:x2], interpolation="nearest")
            ax.set_title(label)
            ax.axis("off")

        plt.tight_layout()
        plt.show()

    # --- error-scaled modes zoomed ---
    for cs in colour_spaces:
        fig, axes = plt.subplots(1, len(modes) + 1, figsize=(20, 5))
        fig.suptitle(
            f"{title_prefix} error-scaled zoomed - {cs.upper()}", fontsize=13
        )

        axes[0].imshow(image[y1:y2, x1:x2], interpolation="nearest")
        axes[0].set_title("Original (zoomed)")
        axes[0].axis("off")

        for ax, mode in zip(axes[1:], modes):
            ax.imshow(
                results["error_scaled"][mode][cs][y1:y2, x1:x2],
                interpolation="nearest",
            )
            ax.set_title(f"error-scaled\n({mode})")
            ax.axis("off")

        plt.tight_layout()
        plt.show()

def plot_error_grid(
    results,
    image,
    colour_spaces,
    alpha,
    modes,
    title_prefix="",
    save_path=None,
    dpi=150,
):
    """
    Big grid like before,mirroring plot_colourspace_comparison_per_strategy, but showing mean absolute error heatmaps instead of reconstructed images.
    Each cell shows the MAE between the dithered result and the original image.
    Rows = strategies, columns = colour spaces.
    """
    
    def _error_map(original, reconstruction):
        return np.abs(original - reconstruction).mean(axis=2)

    strategies = _build_strategies_list(results, alpha, modes)

    n_rows = len(strategies)
    # let's add one more column for the colourbar
    n_cols = len(colour_spaces) + 1

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(4 * len(colour_spaces) + 2, 3.5 * n_rows),
        gridspec_kw={"width_ratios": [0.15] + [1] * len(colour_spaces)},
    )
    fig.suptitle(
        f"{title_prefix} - mean absolute error vs original",
        fontsize=14, fontweight="bold",
    )

    for col, cs in enumerate(colour_spaces, start=1):
        axes[0, col].set_title(cs.upper(), fontsize=11, pad=6)

    # hide the entire colourbar column except the last cell
    for row in range(n_rows - 1):
        axes[row, -1].set_axis_off()

    im_ref = None
    for row, (strategy_name, res_dict) in enumerate(strategies):
        axes[row, 0].set_axis_off()
        axes[row, 0].text(
            0.5, 0.5, strategy_name,
            transform=axes[row, 0].transAxes,
            fontsize=9, va="center", ha="center", rotation=90,
        )

        for col, cs in enumerate(colour_spaces, start=1):
            err = _error_map(image, res_dict[cs])
            im  = axes[row, col].imshow(err, cmap="hot", vmin=0, vmax=0.2)
            axes[row, col].set_xlabel(f"MAE={err.mean():.4f}", fontsize=8)
            axes[row, col].set_xticks([])
            axes[row, col].set_yticks([])
            im_ref = im

    plt.tight_layout(rect=[0, 0, 0.92, 0.97])
    
    # add colourbar manually in the reserved right margin
    cbar_ax = fig.add_axes([0.93, 0.1, 0.015, 0.8])
    fig.colorbar(im_ref, cax=cbar_ax, label="Mean absolute error")

    if save_path is not None:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")

    plt.show()
    return fig

# this function is a convenience wrapper that calls all the above plotting functions in one go
def plot_all(
    results,
    image,
    colour_spaces,
    alpha,
    modes,
    title_prefix="",
    zoom=None,
    save_comparison=None,
    save_error_grid=None,
    dpi=150,
):
    plot_strategy_comparison_per_cs(
        results, image, colour_spaces, alpha, modes, title_prefix
    )
    plot_error_scaled_per_cs(
        results, image, colour_spaces, modes, title_prefix
    )
    plot_colourspace_comparison_per_strategy(
        results, image, colour_spaces, alpha, modes, title_prefix,
        save_path=save_comparison, dpi=dpi,
    )
    plot_error_grid(
        results, image, colour_spaces, alpha, modes,
        title_prefix=title_prefix,
        save_path=save_error_grid,
        dpi=dpi,
    )
    if zoom is not None:
        plot_zoomed_comparison(
            results, image, colour_spaces, alpha, modes, zoom, title_prefix
        )