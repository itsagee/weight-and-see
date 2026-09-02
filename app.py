"""
app.py  –  Dithering GUI (Gradio 6)
──────────────────────────────────────────────────────────────────────────────
Run this from the project root:

    gradio app.py          # auto-reloads on file save
    python app.py          # single run
"""

import io
import sys
import importlib
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import gradio as gr
from PIL import Image as PILImage
from runners.metrics_runner import METRIC_LABELS

# imports for all the dithering functionality
from dithering.io  import load_palette, load_weights
from runners.dithering_runner import run_all_strategies
from runners.dithering_viz    import plot_colourspace_comparison_per_strategy, plot_error_grid
from runners.metrics_runner   import compute_metrics_df, plot_metrics_bars

import dithering.floyd_steinberg     as fs_mod
import dithering.jarvis_judice_ninke as jjn_mod
import dithering.stucki              as st_mod

# tiny fix
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

# setting up some values
ALGO_MAP = {
    "Floyd-Steinberg":     fs_mod,
    "Jarvis-Judice-Ninke": jjn_mod,
    "Stucki":              st_mod,
}
ALGO_PREFIX = {
    "Floyd-Steinberg":     "fs",
    "Jarvis-Judice-Ninke": "jjn",
    "Stucki":              "st",
}
CS_DISPLAY = {
    "RGB":    "rgb",
    "CIELAB": "cielab",
    "CIExyY": "ciexyy",
}
FLAT_STRATEGY_CHOICES = ["Nearest-colour", "Weight-driven", "Weighted-nearest", "Softmax"]
FLAT_KEY_MAP = {
    "Nearest-colour":   "nearest",
    "Weight-driven":    "weights",
    "Weighted-nearest": "combined",
    "Softmax":          "softmax",
}
ERROR_SCALED_MODES  = ["scale", "weighted_target", "confidence"]
ALL_METRIC_CHOICES  = ["PSNR", "SSIM", "CIEDE2000", "S-CIELAB", "LPIPS", "DreamSim"]


# Pure helpers  (no Gradio dependency)
# ───────────────────────────────────────

def resolve_output_dir(output_dir_str):
    """This function resolves the user-supplied output directory to an absolute path, anchored to this script's own location (ROOT) rather than the
    process's current working directory. `gradio app.py` reload mode can run with an unexpected cwd (e.g. a Gradio temp cache folder), which
    otherwise silently redirects relative output paths like './output'.
    """
    
    p = Path((output_dir_str or "./output").strip() or "./output")
    if not p.is_absolute():
        p = (ROOT / p).resolve()
    return p

# this function is converts a matplotlib figure to a PIL image, closing the figure afterwards
def fig_to_pil(fig) -> PILImage.Image:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)
    pil = PILImage.open(buf).copy()
    buf.close()
    plt.close(fig)
    return pil

# this function converts an array to a PIL image, scaling to [0, 255] and converting to uint8
def arr_to_pil(arr) -> PILImage.Image:
    return PILImage.fromarray((np.clip(arr, 0, 1) * 255).astype(np.uint8))

# this function converts a PIL image to a numpy array, scaling to [0, 1] and converting to float32
def get_result_array(results, algo_name, strategy_label, cs_key):
    r   = results[algo_name]
    key = FLAT_KEY_MAP.get(strategy_label)
    if key:
        return r[key][cs_key]
    for mode in ERROR_SCALED_MODES:
        if strategy_label == f"Error-scaled ({mode})":
            return r["error_scaled"][mode][cs_key]
    raise ValueError(f"Unknown strategy label: {strategy_label!r}")

def strategy_matches_filter(label, wanted):
    """Needed because build_selected_strategies() renames Weighted-nearest and softmax to include the alpha value, so a plain 
    exact-match against the checkbox selection would otherwise never match those two strategies.
    """
    if label in wanted:
        return True
    return any(label.startswith(w + " (") for w in wanted)

# this function returns a list of (label, array) tuples for all strategies selected in the metrics panel, for a given algorithm and colour space
def build_selected_strategies(results_for_algo, flat_sel, es_modes_sel, alpha):
    strategies = []
    for label in FLAT_STRATEGY_CHOICES:
        if label not in flat_sel:
            continue
        key   = FLAT_KEY_MAP[label]
        entry = results_for_algo[key]
        if label == "Weighted-nearest":
            label = f"Weighted-nearest (α={alpha})"
        elif label == "Softmax":
            label = f"Softmax (α={alpha})"
        strategies.append((label, entry))
    for mode in es_modes_sel:
        strategies.append(
            (f"Error-scaled ({mode})", results_for_algo["error_scaled"][mode])
        )
    return strategies


# Save / load .npz
# ─────────────────────────────────────────────────────────────────────────────

# this function saves all results arrays to a single .npz file per algorithm, including the original image and metadata (colour spaces, alpha, modes, image name, algorithm name)
def save_results_npz(all_results, image, colour_spaces, alpha, modes,
                     image_name, output_dir):
    import json
    saved = []
    for algo_name, results in all_results.items():
        prefix   = ALGO_PREFIX[algo_name]
        out_path = (Path(output_dir) / image_name
                    / f"{image_name}_{prefix}_results.npz")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        meta = {
            "colour_spaces": colour_spaces,
            "alpha":         alpha,
            "modes":         modes,
            "image_name":    image_name,
            "algo_name":     algo_name,
        }
        arrays = {
            "image":    image,
            "meta_json": np.frombuffer(
                json.dumps(meta).encode("utf-8"), dtype=np.uint8
            ),
        }
        for key in ["nearest", "weights", "combined", "softmax"]:
            for cs in colour_spaces:
                arrays[f"{key}__{cs}"] = results[key][cs]
        for mode in modes:
            for cs in colour_spaces:
                arrays[f"error_scaled__{mode}__{cs}"] = \
                    results["error_scaled"][mode][cs]
        np.savez_compressed(str(out_path), **arrays)
        saved.append(str(out_path))
    return saved

# this function loads all results arrays from a list of .npz files, returning a dict of results per algorithm, the original image, colour spaces, alpha, modes, and image name
def load_results_npz(npz_paths, output_dir="./output"):
    import json
    all_results       = {}
    colour_spaces     = None
    alpha             = None
    modes             = None
    image_name        = None
    image             = None
    colour_spaces_sel = None

    for path in npz_paths:
        data = np.load(path, allow_pickle=False)
        
        # decode metadata, should supports both new (meta_json) and old (meta__*) format
        if "meta_json" in data.files:
            meta     = json.loads(data["meta_json"].tobytes().decode("utf-8"))
            algo_name = meta["algo_name"]
            cs_list   = meta["colour_spaces"]
            a         = float(meta["alpha"])
            m_list    = meta["modes"]
            img_name  = meta["image_name"]
        else:
            algo_name = str(data["meta__algo_name"])
            cs_list   = json.loads(str(data["meta__colour_spaces"]))
            a         = float(data["meta__alpha"])
            m_list    = json.loads(str(data["meta__modes"]))
            img_name  = str(data["meta__image_name"])
        img = data["image"]

        if colour_spaces is None:
            colour_spaces     = cs_list
            alpha             = a
            modes             = m_list
            image_name        = img_name
            image             = img
            colour_spaces_sel = [k for k, v in CS_DISPLAY.items() if v in cs_list]
        else:
            if cs_list != colour_spaces:
                return {}, {}, [], f"❌ Colour spaces differ between files ({path})."
            if img_name != image_name:
                return {}, {}, [], f"❌ Image names differ between files ({path})."

        results = {
            "nearest": {}, "weights": {}, "combined": {}, "softmax": {},
            "error_scaled": {m: {} for m in m_list},
        }
        for key in data.files:
            if key.startswith("meta__") or key == "image":
                continue
            parts = key.split("__")
            if parts[0] in ("nearest", "weights", "combined", "softmax"):
                results[parts[0]][parts[1]] = data[key]
            elif parts[0] == "error_scaled":
                results["error_scaled"][parts[1]][parts[2]] = data[key]
        all_results[algo_name] = results

    if not all_results:
        return {}, {}, [], "❌ No valid .npz files found."

    # we want to show the comparison figures when loading a .npz, so we need to generate them here
    comparison_figures = []
    for algo_name, results in all_results.items():
        fig = plot_colourspace_comparison_per_strategy(
            results       = results,
            image         = image,
            colour_spaces = colour_spaces,
            alpha         = alpha,
            modes         = modes if modes else ["scale"],
            title_prefix  = algo_name,
            save_path     = None,
            dpi           = 150,
        )
        comparison_figures.append((fig_to_pil(fig), algo_name))

    flat_strategies = FLAT_STRATEGY_CHOICES
    results_state = {
        "results":           all_results,
        "image":             image,
        "colour_spaces":     colour_spaces,
        "colour_spaces_sel": colour_spaces_sel,
        "alpha":             alpha,
        "modes":             modes,
        "flat_strategies":   flat_strategies,
        "image_name":        image_name,
        "output_dir":        str(resolve_output_dir(output_dir)),
    }
    # starts empty because it will be filled lazily when user clicks Show Metrics
    metrics_state = {}

    algo_names = list(all_results.keys())
    status = (
        f"Loaded {len(algo_names)} algorithm(s) for '{image_name}': "
        f"{', '.join(algo_names)}\n"
        f"Colour spaces: {colour_spaces}  |  Alpha: {alpha}  |  Modes: {modes}\n"
        f"(p not stored — was set at run time)"
    )
    return results_state, metrics_state, comparison_figures, status


# Core run
# ─────────────────────────────────────────────────────────────────────────────

# biig 'main' function that runs all selected algorithms, colour spaces, and strategies on a given image, palette, and weights file, saving the outputs to disk and returning a results_state dict and metrics_state dict for later use in the GUI
def run_dithering(image_file, palette_path, weights_path,
                  algorithms_sel, colour_spaces_sel,
                  alpha, p,
                  flat_strategies_sel, use_error_scaled, es_modes_sel,
                  output_dir):
    errors = []
    if image_file is None:        errors.append("• Please upload an image.")
    # gr.File returns an object with a .name attribute (the temp path)
    palette_path = palette_path.name if hasattr(palette_path, 'name') else (palette_path or "")
    weights_path = weights_path.name if hasattr(weights_path, 'name') else (weights_path or "")
    if not palette_path:           errors.append("• Palette file is empty.")
    elif not Path(palette_path).exists():
        errors.append(f"• Palette file not found: {palette_path}")
    if not weights_path:           errors.append("• Weights file is empty.")
    elif not Path(weights_path).exists():
        errors.append(f"• Weights file not found: {weights_path}")
    if not algorithms_sel:        errors.append("• Select at least one algorithm.")
    if not colour_spaces_sel:     errors.append("• Select at least one colour space.")
    if not flat_strategies_sel and not use_error_scaled:
        errors.append("• Select at least one strategy.")
    if use_error_scaled and not es_modes_sel:
        errors.append("• Error-scaled enabled but no modes selected.")
    if errors:
        return ("X  " + "\n".join(errors), {}, [], [], {})

    modes_to_run = es_modes_sel if use_error_scaled else []

    try:
        for mod in ALGO_MAP.values():
            importlib.reload(mod)

        image_pil = PILImage.open(image_file).convert("RGB")
        image     = np.array(image_pil).astype(np.float64) / 255.0
        palette   = load_palette(palette_path)
        weights   = load_weights(weights_path)

        image_name    = Path(image_file).stem
        colour_spaces = [CS_DISPLAY[c] for c in colour_spaces_sel]
        resolved_output_dir = resolve_output_dir(output_dir)

        all_results   = {}
        saved_dirs    = []
        n_files_total = 0

        for algo_name in algorithms_sel:
            out_dir = resolved_output_dir / image_name / ALGO_PREFIX[algo_name]
            out_dir.mkdir(parents=True, exist_ok=True)
            all_results[algo_name] = run_all_strategies(
                algo_module   = ALGO_MAP[algo_name],
                image         = image,
                palette       = palette,
                weights       = weights,
                colour_spaces = colour_spaces,
                alpha         = alpha,
                modes         = modes_to_run if modes_to_run else ["scale"],
                p             = p,
                output_dir    = str(out_dir),
                image_name    = image_name,
                prefix        = ALGO_PREFIX[algo_name],
            )
            saved_dirs.append(str(out_dir))
            n_files_total += len(colour_spaces) * (
                len(flat_strategies_sel) + len(modes_to_run)
            )

        # save arrays immediately after all_results is complete
        save_results_npz(
            all_results, image, colour_spaces, alpha, modes_to_run,
            image_name, str(resolved_output_dir),
        )

        # comparison figures
        comparison_figures = []
        for algo_name in algorithms_sel:
            fig_save_path = str(
                resolved_output_dir / image_name / ALGO_PREFIX[algo_name]
                / f"{image_name}_{ALGO_PREFIX[algo_name]}_strategy_comparison.png"
            )
            error_grid_path = str(
                resolved_output_dir / image_name / ALGO_PREFIX[algo_name]
                / f"{image_name}_{ALGO_PREFIX[algo_name]}_error_grid.png"
            )
            fig = plot_colourspace_comparison_per_strategy(
                results       = all_results[algo_name],
                image         = image,
                colour_spaces = colour_spaces,
                alpha         = alpha,
                modes         = modes_to_run,
                title_prefix  = algo_name,
                save_path     = fig_save_path,
                dpi           = 150,
            )
            comparison_figures.append((fig_to_pil(fig), algo_name))

            # error grid — new addition
            fig_err = plot_error_grid(
                results       = all_results[algo_name],
                image         = image,
                colour_spaces = colour_spaces,
                alpha         = alpha,
                modes         = modes_to_run,
                title_prefix  = algo_name,
                save_path     = error_grid_path,
                dpi           = 150,
            )
            comparison_figures.append((fig_to_pil(fig_err), f"{algo_name} — error grid"))

        results_state = {
            "results":           all_results,
            "image":             image,
            "colour_spaces":     colour_spaces,
            "colour_spaces_sel": colour_spaces_sel,
            "alpha":             alpha,
            "modes":             modes_to_run,
            "flat_strategies":   flat_strategies_sel,
            "image_name":        image_name,
            "output_dir":        str(resolved_output_dir),
        }

        metrics_state = {}
        for algo_name in algorithms_sel:
            strats = build_selected_strategies(
                all_results[algo_name], FLAT_STRATEGY_CHOICES, modes_to_run, alpha
            )
            df = compute_metrics_df(strats, image, colour_spaces)
            df.insert(0, "Algorithm", algo_name)
            metrics_state[algo_name] = df

        dirs_str = "\n".join(f"   {d}" for d in saved_dirs)
        status   = (
            f":) Done!  {n_files_total} images saved across "
            f"{len(algorithms_sel)} algorithm(s):\n{dirs_str}"
        )
        return (status, results_state, comparison_figures, [], metrics_state)

    except Exception:
        return (f":( Error:\n{traceback.format_exc()}", {}, [], [], {})


# Viewer helpers
# ─────────────────────────────────────────────────────────────────────────────

# this function builds the list of available strategy choices for the metrics panel, based on the selected flat strategies and error-scaled modes
def build_strategy_choices(flat_sel, use_es, es_modes):
    choices = list(flat_sel or [])
    if use_es:
        choices += [f"Error-scaled ({m})" for m in (es_modes or [])]
    return gr.update(choices=choices, value=choices[0] if choices else None)

# this function builds the list of available strategy choices for the metrics panel, based on a loaded results_state dict
def strategy_choices_from_state(rs):
    """Derive available strategy choices from a loaded results_state dict."""
    if not rs or "results" not in rs:
        return gr.update(choices=[], value=None)
    
    # see compute_metrics_state: all four flat strategies are always saved, regardless of what was checked on the Run tab, so always offer all four
    flat  = FLAT_STRATEGY_CHOICES
    modes = rs.get("modes", [])
    choices = list(flat)
    choices += [f"Error-scaled ({m})" for m in modes]
    return gr.update(choices=choices, value=choices[0] if choices else None)

# this function adds a new tile to the gallery, given the current gallery_state, results_state, and the selected algorithm, colour space, and strategy
def add_tile(gallery_state, results_state, algo_sel, cs_sel, strategy_sel):
    if not results_state:
        return gallery_state, "Run the pipeline first."
    if not algo_sel or not cs_sel or not strategy_sel:
        return gallery_state, "Pick an algorithm, colour space, and strategy."
    cs_key = CS_DISPLAY.get(cs_sel, cs_sel.lower())
    
    try:
        arr = get_result_array(results_state["results"], algo_sel, strategy_sel, cs_key)
    except (KeyError, ValueError) as e:
        return gallery_state, f"Could not retrieve image: {e}"
    caption = f"{algo_sel} | {cs_sel} | {strategy_sel}"
    
    if caption in [c for _, c in gallery_state]:
        return gallery_state, f"Already shown: {caption}"
    return gallery_state + [(arr_to_pil(arr), caption)], f"Added: {caption}"

# this function removes the last tile from the gallery
def remove_last_tile(gs):
    if not gs:
        return gs, "Gallery is already empty."
    return gs[:-1], f"Removed last tile. ({len(gs)-1} remaining)"

# this function clears the gallery
def clear_gallery(gs):
    return [], "Gallery cleared."


# Metrics helpers
# ─────────────────────────────────────────────────────────────────────────────

def compute_metrics_state(results_state):
    """This function lazily computes per-algorithm metrics dataframes from a results_state dict (used both after Run and after Load, 
    so metrics are only computed once the user actually asks for them via 'Show metrics' / 'Save CSV').
    """
    
    all_results = results_state.get("results", {})
    if not all_results:
        return {}
    image             = results_state.get("image")
    colour_spaces     = results_state.get("colour_spaces", [])
    flat_strategies   = FLAT_STRATEGY_CHOICES
    modes_to_run      = results_state.get("modes", [])
    alpha             = results_state.get("alpha", 0.5)

    metrics_state = {}
    for algo_name, results_for_algo in all_results.items():
        strats = build_selected_strategies(
            results_for_algo, flat_strategies, modes_to_run, alpha
        )
        df = compute_metrics_df(strats, image, colour_spaces)
        df.insert(0, "Algorithm", algo_name)
        metrics_state[algo_name] = df
    return metrics_state

# this function shows and auto-saves the metrics dataframes to a single CSV file, returning the path of the saved file
def show_metrics_ui(metrics_state, results_state, metrics_sel, metrics_cs_sel,
                    metrics_algo_sel, metrics_flat_sel,
                    metrics_use_es, metrics_es_sel,
                    show_combined, per_algo_charts, show_all_strategies_chart):
    if not metrics_state:
        metrics_state = compute_metrics_state(results_state or {})
    if not metrics_state:
        return "Run or load results first.", pd.DataFrame(), []
    if not metrics_sel:
        return "Select at least one metric.", pd.DataFrame(), []
    if not metrics_cs_sel:
        return "Select at least one colour space.", pd.DataFrame(), []
    if not metrics_algo_sel:
        return "Select at least one algorithm.", pd.DataFrame(), []

    strat_filter = set(metrics_flat_sel or [])
    if metrics_use_es:
        for mode in (metrics_es_sel or []):
            strat_filter.add(f"Error-scaled ({mode})")
    if not strat_filter:
        return "Select at least one strategy.", pd.DataFrame(), []

    cs_upper = [c.upper() for c in metrics_cs_sel]
    frames   = []
    for algo_name in metrics_algo_sel:
        df = metrics_state.get(algo_name)
        if df is None:
            continue
        cols = ["Algorithm", "Strategy", "Colour space"] + [
            m for m in metrics_sel if m in df.columns
        ]
        df_f = df[cols]
        df_f = df_f[df_f["Colour space"].isin(cs_upper)]
        df_f = df_f[df_f["Strategy"].apply(lambda s: strategy_matches_filter(s, strat_filter))]
        frames.append((algo_name, df_f.reset_index(drop=True)))

    if not frames:
        return "No data matched the selected filters.", pd.DataFrame(), []

    all_frames = [df for _, df in frames]
    display_df = pd.concat(all_frames, ignore_index=True)

    bar_figs = []

    # --- per-algorithm bar charts ---
    if per_algo_charts:
        for algo_name, df_f in frames:
            plot_metrics_bars(df_f, metrics=metrics_sel, colour_spaces=cs_upper)
            fig = plt.gcf()
            fig.suptitle(f"Metrics — {algo_name}",
                         fontsize=13, fontweight="bold", y=1.01)
            bar_figs.append((fig_to_pil(fig), f"Metrics — {algo_name}"))

    # --- combined bar chart (only meaningful with >1 algorithm) ---
    if show_combined and len(frames) > 1:
        algo_abbrev = {
            "Floyd-Steinberg": "FS",
            "Jarvis-Judice-Ninke": "JJN",
            "Stucki": "ST",
        }
        colours = plt.cm.tab10.colors

        # build a labelled + coloured dataframe for the combined chart
        combined_frames = []
        colour_map = {}
        for ai, (algo_name, df_f) in enumerate(frames):
            abbrev = algo_abbrev.get(algo_name, algo_name)
            df_copy = df_f.copy()
            df_copy["Strategy"] = df_copy["Strategy"].apply(
                lambda s: f"{abbrev} · {s}"
            )
            combined_frames.append(df_copy)
            
            # assign a colour to every prefixed strategy from this algo
            for s in df_copy["Strategy"].unique():
                colour_map[s] = colours[ai % len(colours)]

        combined = pd.concat(combined_frames, ignore_index=True)

        # custom plot so we can apply per-bar colours
        n_metrics = len(metrics_sel)
        n_cs      = len(cs_upper)
        fig, axes = plt.subplots(
            n_metrics, n_cs,
            figsize=(6 * n_cs, 4 * n_metrics),
            sharey="row",
        )
        # normalise axes to 2D
        if n_metrics == 1: axes = [axes]
        if n_cs      == 1: axes = [[row] for row in axes]

        fig.suptitle("Metrics — all algorithms combined",
                    fontsize=13, fontweight="bold")

        for row, metric in enumerate(metrics_sel):
            for col, cs in enumerate(cs_upper):
                ax     = axes[row][col]
                subset = combined[combined["Colour space"] == cs].reset_index(drop=True)
                strats = subset["Strategy"].tolist()
                vals   = subset[metric].tolist() if metric in subset.columns else []
                bar_colours = [colour_map.get(s, "grey") for s in strats]

                ax.bar(range(len(strats)), vals, color=bar_colours)

                if row == 0:
                    ax.set_title(cs, fontsize=12, fontweight="bold")
                if col == 0:
                    ax.set_ylabel(METRIC_LABELS.get(metric, metric), fontsize=10)

                ax.set_xticks(range(len(strats)))
                ax.set_xticklabels(strats, rotation=35, ha="right", fontsize=7)
                ax.tick_params(axis="y", labelsize=8)

        # legend to show one entry per algorithm
        from matplotlib.patches import Patch
        legend_handles = [
            Patch(color=colours[ai % len(colours)],
                label=f"{algo_abbrev.get(n, n)} — {n}")
            for ai, (n, _) in enumerate(frames)
        ]
        fig.legend(
            handles=legend_handles,
            loc="lower center",
            ncol=len(frames),
            fontsize=9,
            bbox_to_anchor=(0.5, -0.02),
        )
        plt.tight_layout(rect=[0, 0.04, 1, 0.97])
        bar_figs.append((fig_to_pil(fig), "Metrics — combined"))

    # --- all-strategies overview (grouped bars per metric) ---
    if show_all_strategies_chart:
        all_df    = pd.concat(all_frames, ignore_index=True)
        algo_list = all_df["Algorithm"].unique().tolist()
        colours   = plt.cm.tab10.colors
        for metric in metrics_sel:
            if metric not in all_df.columns:
                continue
            fig, axes = plt.subplots(
                1, len(cs_upper),
                figsize=(max(8, len(strat_filter) * len(algo_list) * 0.9 + 2), 5),
                sharey=False,
            )
            if len(cs_upper) == 1:
                axes = [axes]
            fig.suptitle(f"All strategies — {metric}",
                         fontsize=13, fontweight="bold")
            for ax, cs in zip(axes, cs_upper):
                subset    = all_df[all_df["Colour space"] == cs]
                strats    = subset["Strategy"].unique().tolist()
                n_algos   = len(algo_list)
                bar_width = 0.8 / max(n_algos, 1)
                for ai, algo in enumerate(algo_list):
                    a_sub = subset[subset["Algorithm"] == algo]
                    vals  = [
                        float(a_sub[a_sub["Strategy"] == s][metric].iloc[0])
                        if not a_sub[a_sub["Strategy"] == s].empty else 0.0
                        for s in strats
                    ]
                    xs = [i + ai * bar_width for i in range(len(strats))]
                    ax.bar(xs, vals, width=bar_width,
                           color=colours[ai % len(colours)], label=algo)
                ax.set_title(cs, fontsize=11)
                ax.set_xticks(
                    [i + bar_width * (n_algos - 1) / 2 for i in range(len(strats))]
                )
                ax.set_xticklabels(strats, rotation=35, ha="right", fontsize=8)
                ax.set_ylabel(metric)
                ax.legend(fontsize=7)
            plt.tight_layout()
            bar_figs.append((fig_to_pil(fig), f"All strategies — {metric}"))

    n_rows = len(display_df)
    status = (
        f"Showing {n_rows} rows · {len(metrics_algo_sel)} algorithm(s) · "
        f"{len(strat_filter)} strateg{'y' if len(strat_filter)==1 else 'ies'} · "
        f"{len(metrics_sel)} metric(s) · {len(cs_upper)} colour space(s)."
    )
    
    # --- auto-save to output/<image_name>/metrics/ ---
    def slugify(text):
        return text.strip().lower().replace(" ", "_").replace("/", "-")

    try:
        image_name = results_state.get("image_name", "unknown")
        base_dir   = results_state.get("output_dir", ".")
        save_dir   = Path(base_dir) / image_name / "metrics"
        save_dir.mkdir(parents=True, exist_ok=True)

        # full metrics CSV
        csv_path = save_dir / f"{image_name}-metrics_all.csv"
        display_df.to_csv(csv_path, index=False)

        # bar chart PNGs - named "{image}-{scope}[-{metric}].png"
        # e.g. "Metrics - Floyd-Steinberg"  -> "girl-floyd-steinberg.png"
        #      "Metrics - combined"         -> "girl-combined.png"
        #      "All strategies — CIEDE2000" -> "girl-all_strategies-ciede2000.png"
        for fig_pil, fig_label in bar_figs:
            if " — " in fig_label:
                prefix, suffix = fig_label.split(" — ", 1)
            else:
                prefix, suffix = fig_label, ""
            prefix_slug = slugify(prefix)
            suffix_slug = slugify(suffix)

            if prefix_slug == "metrics":
                fname = f"{image_name}-{suffix_slug}.png"
            elif suffix_slug:
                fname = f"{image_name}-{prefix_slug}-{suffix_slug}.png"
            else:
                fname = f"{image_name}-{prefix_slug}.png"

            fig_pil.save(save_dir / fname)

        status += f"\n:) Auto-saved to {save_dir}"
    except Exception as e:
        status += f"\n-.-  Could not auto-save: {e}"

    return status, display_df, bar_figs

# this function saves the metrics dataframes to a single CSV file, returning the path of the saved file
def save_csv_ui(metrics_state, results_state, metrics_sel, metrics_cs_sel,
                metrics_algo_sel, metrics_flat_sel,
                metrics_use_es, metrics_es_sel, output_dir):
    if not metrics_state:
        metrics_state = compute_metrics_state(results_state or {})
    if not metrics_state:
        return "Run or load results first.", gr.update(visible=False)
    
    strat_filter = set(metrics_flat_sel or [])
    if metrics_use_es:
        for mode in (metrics_es_sel or []):
            strat_filter.add(f"Error-scaled ({mode})")
    cs_upper = [c.upper() for c in metrics_cs_sel]
    frames   = []
    for algo_name in (metrics_algo_sel or list(metrics_state.keys())):
        df = metrics_state.get(algo_name)
        if df is None:
            continue
        # only keep selected metrics
        cols = ["Algorithm", "Strategy", "Colour space"] + [
            m for m in metrics_sel if m in df.columns
        ]
        df_f = df[cols]
        df_f = df_f[df_f["Colour space"].isin(cs_upper)]
        if strat_filter:
            df_f = df_f[df_f["Strategy"].apply(lambda s: strategy_matches_filter(s, strat_filter))]
        frames.append(df_f)
    if not frames:
        return "No data matched the filters.", gr.update(visible=False)
    combined = pd.concat(frames, ignore_index=True)
    image_name = results_state.get("image_name", "unknown")
    save_dir = resolve_output_dir(output_dir) / image_name / "metrics"
    save_dir.mkdir(parents=True, exist_ok=True)
    out_path = save_dir / f"{image_name}-metrics_all.csv"
    combined.to_csv(out_path, index=False)
    return f":)  Saved to {out_path}", gr.update(value=str(out_path), visible=True)

# UI  — all components defined top-to-bottom, wiring at the very end
# ─────────────────────────────────────────────────────────────────────────────

CSS = """
.gradio-container { max-width: 100% !important; }
footer { display: none !important; }
"""

with gr.Blocks(title="Dithering GUI") as demo:

    gr.Markdown("# Dithering Pipeline")

    # shared state
    results_state_comp    = gr.State({})
    gallery_state_comp    = gr.State([])
    comparison_figs_state = gr.State([])
    metrics_state_comp    = gr.State({})

    # ── top row: inputs (narrow) + status/load (wide) ─────────────────────────
    with gr.Row():
        # LEFT — run inputs
        with gr.Column(scale=1, min_width=260):
            gr.Markdown("### Input files")
            image_input   = gr.Image(label="Image", type="filepath")
            palette_input = gr.File(
                label="Palette file (.js)",
                file_types=[".js"],
            )
            weights_input = gr.File(
                label="Weights file (.js)",
                file_types=[".js"],
            )
            output_input = gr.Textbox(label="Output directory", value="./output")

            gr.Markdown("### Algorithms")
            algo_input = gr.CheckboxGroup(
                choices=list(ALGO_MAP.keys()),
                value=["Floyd-Steinberg"],
                label="",
            )

            gr.Markdown("### Colour spaces")
            cs_input = gr.CheckboxGroup(
                choices=list(CS_DISPLAY.keys()),
                value=list(CS_DISPLAY.keys()),
                label="",
            )

            gr.Markdown("### Hyperparameters")
            alpha_input = gr.Slider(0.0, 1.0, step=0.05, value=0.5, label="Alpha (α)")
            p_input     = gr.Slider(0.5, 5.0, step=0.5,  value=2.0, label="p (confidence)")

            gr.Markdown("### Strategies")
            flat_strategies_input = gr.CheckboxGroup(
                choices=FLAT_STRATEGY_CHOICES,
                value=FLAT_STRATEGY_CHOICES,
                label="",
            )
            use_error_scaled_input = gr.Checkbox(value=True, label="Include error-scaled")
            with gr.Group(visible=True) as es_modes_group:
                es_modes_input = gr.CheckboxGroup(
                    choices=ERROR_SCALED_MODES,
                    value=ERROR_SCALED_MODES,
                    label="Error-scaled modes",
                )

            run_btn = gr.Button("▶  Run", variant="primary", size="lg")

        # RIGHT — status + load
        with gr.Column(scale=3):
            status_out = gr.Textbox(label="Status", lines=3, interactive=False)

            gr.Markdown("---")
            gr.Markdown(
                "### Load saved results\n"
                "Select `.npz` files from a previous run (one per algorithm, "
                "found in `output/<image>/<algo>/`) to skip recomputation."
            )
            with gr.Row():
                load_files_input = gr.File(
                    label="Select .npz result file(s)",
                    file_count="multiple",
                    file_types=[".npz"],
                    scale=3,
                )
                load_btn = gr.Button("Load  📂", variant="secondary", scale=1)
            load_status = gr.Textbox(
                label="", lines=2, interactive=False, show_label=False
            )

    # ── comparison figures ────────────────────────────────────────────────────
    gr.Markdown("---")
    gr.Markdown(
        "### All-strategies comparison  "
        "*(one figure per algorithm, auto-saved to output directory)*"
    )
    comparison_gallery = gr.Gallery(
        label="", object_fit="contain", show_label=False
    )

    # ── custom viewer ─────────────────────────────────────────────────────────
    gr.Markdown("---")
    gr.Markdown(
        "### Custom viewer\n"
        "Pick any combination and add it to the gallery. No re-running needed."
    )
    with gr.Row():
        viewer_algo     = gr.Dropdown(choices=[], label="Algorithm",    scale=2)
        viewer_cs       = gr.Dropdown(choices=[], label="Colour space", scale=2)
        viewer_strategy = gr.Dropdown(choices=[], label="Strategy",     scale=3)
    with gr.Row():
        add_btn    = gr.Button("➕  Add",        variant="secondary")
        remove_btn = gr.Button("↩  Remove last", variant="secondary")
        clear_btn  = gr.Button("🗑  Clear all",  variant="stop")
    viewer_status  = gr.Textbox(
        label="", lines=1, interactive=False, show_label=False
    )
    custom_gallery = gr.Gallery(
        label="", columns=3, object_fit="contain", show_label=False
    )

    # ── metrics ───────────────────────────────────────────────────────────────
    gr.Markdown("---")
    gr.Markdown("### Metrics")
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("**What to measure**")
            metrics_sel = gr.CheckboxGroup(
                choices=ALL_METRIC_CHOICES,
                value=["PSNR", "SSIM", "CIEDE2000"],
                label="Metrics",
            )
            metrics_cs_sel = gr.CheckboxGroup(
                choices=list(CS_DISPLAY.keys()),
                value=list(CS_DISPLAY.keys()),
                label="Colour spaces",
            )
        with gr.Column(scale=1):
            gr.Markdown("**Which results to include**")
            metrics_algo_sel = gr.CheckboxGroup(
                choices=[], value=[], label="Algorithms"
            )
            metrics_flat_sel = gr.CheckboxGroup(
                choices=FLAT_STRATEGY_CHOICES,
                value=FLAT_STRATEGY_CHOICES,
                label="Strategies",
            )
            metrics_use_es = gr.Checkbox(value=True, label="Include error-scaled")
            with gr.Group(visible=True) as metrics_es_group:
                metrics_es_sel = gr.CheckboxGroup(
                    choices=ERROR_SCALED_MODES,
                    value=ERROR_SCALED_MODES,
                    label="Error-scaled modes",
                )
        with gr.Column(scale=1):
            gr.Markdown("**Display options**")
            gr.Markdown(
                "<small>Select which chart types to generate. "
                "Each is independent — mix and match as needed.</small>"
            )
            per_algo_charts = gr.Checkbox(
                value=True,
                label="Per-algorithm bar chart",
                info="One bar chart per selected algorithm, with strategies as bars."
            )
            show_combined = gr.Checkbox(
                value=False,
                label="Combined bar chart",
                info="All algorithms on one chart. Only available when >1 algorithm is selected.",
                interactive=False,
            )
            show_all_strategies_chart = gr.Checkbox(
                value=True,
                label="All-strategies overview",
                info="Grouped bars comparing all strategies side by side, per metric."
            )

    with gr.Row():
        metrics_btn  = gr.Button("Show metrics  📊", variant="primary")
        save_csv_btn = gr.Button("Save CSV  💾",     variant="secondary")
    metrics_status  = gr.Textbox(
        label="", lines=1, interactive=False, show_label=False
    )
    metrics_df_out  = gr.Dataframe(
        label="Metrics table", interactive=False, wrap=True
    )
    metrics_gallery_out = gr.Gallery(
        label="Bar charts", columns=1, object_fit="contain", show_label=True
    )
    csv_file_out = gr.File(label="Download CSV", visible=False)

    # ── wiring ────────────────────────────────────────────────────────────────

    # show/hide es modes panel
    use_error_scaled_input.change(
        fn=lambda v: gr.update(visible=v),
        inputs=[use_error_scaled_input],
        outputs=[es_modes_group],
    )
    metrics_use_es.change(
        fn=lambda v: gr.update(visible=v),
        inputs=[metrics_use_es],
        outputs=[metrics_es_group],
    )
    
    # disable combined chart when only 1 algorithm is selected
    metrics_algo_sel.change(
        fn=lambda algos: gr.update(
            interactive=len(algos) > 1,
            value=False if len(algos) <= 1 else None,
        ),
        inputs=[metrics_algo_sel],
        outputs=[show_combined],
    )

    # keep viewer strategy dropdown in sync with strategy toggles
    for inp in [flat_strategies_input, use_error_scaled_input, es_modes_input]:
        inp.change(
            fn=build_strategy_choices,
            inputs=[flat_strategies_input, use_error_scaled_input, es_modes_input],
            outputs=[viewer_strategy],
        )

    # Run button
    run_btn.click(
        fn=run_dithering,
        inputs=[
            image_input, palette_input, weights_input,
            algo_input, cs_input, alpha_input, p_input,
            flat_strategies_input, use_error_scaled_input, es_modes_input,
            output_input,
        ],
        outputs=[
            status_out,
            results_state_comp, comparison_figs_state,
            gallery_state_comp, metrics_state_comp,
        ],
    ).then(
        fn=lambda figs: figs,
        inputs=[comparison_figs_state],
        outputs=[comparison_gallery],
    ).then(
        fn=lambda s: s,
        inputs=[gallery_state_comp],
        outputs=[custom_gallery],
    ).then(
        fn=build_strategy_choices,
        inputs=[flat_strategies_input, use_error_scaled_input, es_modes_input],
        outputs=[viewer_strategy],
    ).then(
        fn=lambda rs: gr.update(
            choices=list(rs.get("results", {}).keys()),
            value=list(rs.get("results", {}).keys()),
        ),
        inputs=[results_state_comp],
        outputs=[viewer_algo],
    ).then(
        fn=lambda rs: gr.update(
            choices=rs.get("colour_spaces_sel", []),
            value=(rs.get("colour_spaces_sel") or [None])[0],
        ),
        inputs=[results_state_comp],
        outputs=[viewer_cs],
    ).then(
        fn=lambda rs: gr.update(
            choices=list(rs.get("results", {}).keys()),
            value=list(rs.get("results", {}).keys()),
        ),
        inputs=[results_state_comp],
        outputs=[metrics_algo_sel],
    ).then(
        fn=lambda flat_sel: gr.update(choices=flat_sel, value=flat_sel),
        inputs=[flat_strategies_input],
        outputs=[metrics_flat_sel],
    ).then(
        fn=lambda v: gr.update(value=v),
        inputs=[use_error_scaled_input],
        outputs=[metrics_use_es],
    ).then(
        fn=lambda modes: gr.update(choices=modes, value=modes),
        inputs=[es_modes_input],
        outputs=[metrics_es_sel],
    ).then(
        fn=lambda rs: gr.update(
            interactive=len(rs.get("results", {})) > 1,
            value=False if len(rs.get("results", {})) <= 1 else None,
        ),
        inputs=[results_state_comp],
        outputs=[show_combined],
    )

    # Load button
    def load_results_ui(files, output_dir):
        if not files:
            return ("No files selected.", {}, [], [], {})
        paths = [f.name for f in files]
        rs, ms, figs, status = load_results_npz(paths, output_dir)
        if not rs:
            return (status, {}, [], [], {})
        return (status, rs, figs, [], ms)

    load_btn.click(
        fn=load_results_ui,
        inputs=[load_files_input, output_input],
        outputs=[
            load_status,
            results_state_comp, comparison_figs_state,
            gallery_state_comp, metrics_state_comp,
        ],
    ).then(
        fn=lambda figs: figs,
        inputs=[comparison_figs_state],
        outputs=[comparison_gallery],
    ).then(
        fn=lambda s: s,
        inputs=[gallery_state_comp],
        outputs=[custom_gallery],
    ).then(
        fn=lambda rs: gr.update(
            choices=list(rs.get("results", {}).keys()),
            value=list(rs.get("results", {}).keys()),
        ),
        inputs=[results_state_comp],
        outputs=[viewer_algo],
    ).then(
        fn=lambda rs: gr.update(
            choices=rs.get("colour_spaces_sel", []),
            value=(rs.get("colour_spaces_sel") or [None])[0],
        ),
        inputs=[results_state_comp],
        outputs=[viewer_cs],
    ).then(
        fn=lambda rs: gr.update(
            choices=list(rs.get("results", {}).keys()),
            value=list(rs.get("results", {}).keys()),
        ),
        inputs=[results_state_comp],
        outputs=[metrics_algo_sel],
    ).then(
        fn=lambda rs: gr.update(
            choices=rs.get("colour_spaces_sel", []),
            value=rs.get("colour_spaces_sel", []),
        ),
        inputs=[results_state_comp],
        outputs=[metrics_cs_sel],
    ).then(
        fn=strategy_choices_from_state,
        inputs=[results_state_comp],
        outputs=[viewer_strategy],
    ).then(
        fn=lambda rs: gr.update(
            interactive=len(rs.get("results", {})) > 1,
            value=False if len(rs.get("results", {})) <= 1 else None,
        ),
        inputs=[results_state_comp],
        outputs=[show_combined],
    )

    # Viewer buttons
    add_btn.click(
        fn=add_tile,
        inputs=[
            gallery_state_comp, results_state_comp,
            viewer_algo, viewer_cs, viewer_strategy,
        ],
        outputs=[gallery_state_comp, viewer_status],
    ).then(
        fn=lambda s: s, inputs=[gallery_state_comp], outputs=[custom_gallery]
    )
    remove_btn.click(
        fn=remove_last_tile,
        inputs=[gallery_state_comp],
        outputs=[gallery_state_comp, viewer_status],
    ).then(
        fn=lambda s: s, inputs=[gallery_state_comp], outputs=[custom_gallery]
    )
    clear_btn.click(
        fn=clear_gallery,
        inputs=[gallery_state_comp],
        outputs=[gallery_state_comp, viewer_status],
    ).then(
        fn=lambda s: s, inputs=[gallery_state_comp], outputs=[custom_gallery]
    )

    # Metrics buttons
    metrics_btn.click(
        fn=show_metrics_ui,
        inputs=[
            metrics_state_comp, results_state_comp,
            metrics_sel, metrics_cs_sel,
            metrics_algo_sel, metrics_flat_sel,
            metrics_use_es, metrics_es_sel,
            show_combined, per_algo_charts, show_all_strategies_chart,
        ],
        outputs=[metrics_status, metrics_df_out, metrics_gallery_out],
    )
    save_csv_btn.click(
        fn=save_csv_ui,
        inputs=[
            metrics_state_comp, results_state_comp,
            metrics_sel, metrics_cs_sel,
            metrics_algo_sel, metrics_flat_sel,
            metrics_use_es, metrics_es_sel,
            output_input,
        ],
        outputs=[metrics_status, csv_file_out],
    )

if __name__ == "__main__":
    demo.launch(inbrowser=True, theme=gr.themes.Default(), css=CSS)