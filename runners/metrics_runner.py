"""
metrics/runner.py
~~~~~~~~~~~~~~~~~
Helpers for computing all quantitative metrics across strategies / colour spaces and visualising the results.

Typical notebook usage
----------------------
    from metrics.runner import compute_metrics_df, plot_metrics_bars

    # ``all_strategies`` is the list produced by _build_strategies_list in viz.py,
    # or you can assemble it manually:
    all_strategies = [
        ("Nearest-colour",   results_fs["nearest"]),
        ("Weight-driven",    results_fs["weights"]),
        ...
    ]

    df = compute_metrics_df(all_strategies, image, colour_spaces)
    display(df)

    plot_metrics_bars(df)
"""

import pandas as pd
import matplotlib.pyplot as plt

from metrics.psnr      import compute_psnr
from metrics.ssim      import compute_ssim
from metrics.ciede2000 import compute_ciede2000
from metrics.s_cielab  import compute_scielab
from metrics.lpips     import compute_lpips
from metrics.dreamsim  import compute_dreamsim

# Default pixels-per-degree for S-CIELAB; override via compute_metrics_df kwarg
_DEFAULT_PPD = 60.0

# Metric display labels (used by plot_metrics_bars)
METRIC_LABELS: dict[str, str] = {
    "PSNR":      "PSNR (dB) ↑",
    "SSIM":      "SSIM ↑",
    "CIEDE2000": "CIEDE2000 ↓",
    "S-CIELAB":  "S-CIELAB ↓",
    "LPIPS":     "LPIPS ↓",
    "DreamSim":  "DreamSim ↓",
}

# ---------------------------------------------------------------------------
# Core evaluation helper
# ---------------------------------------------------------------------------

def _evaluate_one(original, reconstruction, pixels_per_degree: float) -> dict:
    """This function returns all six metric scores for a single (original, reconstruction) pair."""
    return {
        "PSNR":      compute_psnr(original, reconstruction),
        "SSIM":      compute_ssim(original, reconstruction),
        "CIEDE2000": compute_ciede2000(original, reconstruction),
        "S-CIELAB":  compute_scielab(original, reconstruction, pixels_per_degree=pixels_per_degree),
        "LPIPS":     compute_lpips(original, reconstruction),
        "DreamSim":  compute_dreamsim(original, reconstruction),
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_metrics_df(strategies: list[tuple[str, dict]], image, colour_spaces: list[str], pixels_per_degree: float = _DEFAULT_PPD,) -> pd.DataFrame:
    """
    This function evaluates all strategies × colour spaces and return a tidy DataFrame.

    Parameters
    ----------
    strategies : list of (strategy_name, results_dict) pairs.
                 Each ``results_dict`` maps colour-space string → H×W×3 array.
                 Construct this from a ``results`` dict returned by
                 ``dithering.runner.run_all_strategies`` like so::

                     from dithering.viz import _build_strategies_list
                     strategies = _build_strategies_list(results, alpha, modes)

    image            : H×W×3 float32 original image
    colour_spaces    : list of colour-space strings, e.g. ['rgb', 'cielab', 'ciexyy']
    pixels_per_degree: viewing-distance parameter for S-CIELAB (default 60)

    Returns
    -------
    pd.DataFrame with columns:
        Strategy, Colour space, PSNR, SSIM, CIEDE2000, S-CIELAB, LPIPS, DreamSim
    """
    rows = []
    for strategy_name, results_dict in strategies:
        for cs in colour_spaces:
            scores = _evaluate_one(image, results_dict[cs], pixels_per_degree)
            rows.append({
                "Strategy":     strategy_name,
                "Colour space": cs.upper(),
                **scores,
            })
    return pd.DataFrame(rows)


def plot_metrics_bars(df: pd.DataFrame, metrics: list[str] | None = None, colour_spaces: list[str] | None = None) -> None:
    """
    Now to be able to make a bar-chart grid: one row per metric, one column per colour space (for the big summary figures).

    Parameters
    ----------
    df            : DataFrame as returned by ``compute_metrics_df``
    metrics       : subset of metric columns to plot; defaults to all six
    colour_spaces : colour-space labels to include (must match 'Colour space'
                    column values); defaults to all present in df
    """
    if metrics is None:
        metrics = list(METRIC_LABELS.keys())
    if colour_spaces is None:
        colour_spaces = df["Colour space"].unique().tolist()

    n_rows = len(metrics)
    n_cols = len(colour_spaces)

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(6 * n_cols, 4 * n_rows),
        sharey="row",
    )

    # Ensure axes is always 2-D
    if n_rows == 1:
        axes = [axes]
    if n_cols == 1:
        axes = [[row] for row in axes]

    fig.suptitle(
        "Metric comparison across strategies and colour spaces",
        fontsize=14,
        fontweight="bold",
    )

    colours = plt.cm.tab10.colors

    for row, metric in enumerate(metrics):
        for col, cs in enumerate(colour_spaces):
            ax = axes[row][col]
            subset = df[df["Colour space"] == cs].reset_index(drop=True)

            ax.bar(
                range(len(subset)),
                subset[metric],
                color=colours[: len(subset)],
            )

            if row == 0:
                ax.set_title(cs, fontsize=12, fontweight="bold")

            if col == 0:
                ax.set_ylabel(METRIC_LABELS.get(metric, metric), fontsize=10)

            ax.set_xticks(range(len(subset)))
            ax.set_xticklabels(
                subset["Strategy"], rotation=35, ha="right", fontsize=8
            )
            ax.tick_params(axis="y", labelsize=8)

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.show()


def build_strategies_list(results: dict, alpha: float, modes: list[str],) -> list[tuple[str, dict]]:
    """
    This function converts a ``run_all_strategies`` results dict into the ``[(label, per-cs-dict), …]`` format expected by ``compute_metrics_df``.

    This mirrors ``dithering.viz._build_strategies_list`` so the metrics runner has no dependency on the viz module.
    """
    
    strategies = [
        ("Nearest-colour",                results["nearest"]),
        ("Weight-driven",                 results["weights"]),
        (f"Weighted-nearest (α={alpha})", results["combined"]),
        (f"Softmax (α={alpha})",           results["softmax"]),
    ]
    for mode in modes:
        strategies.append((f"Error-scaled ({mode})", results["error_scaled"][mode]))
    
    return strategies
