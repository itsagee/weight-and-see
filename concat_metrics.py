"""
concat_metrics.py
─────────────────
This scipt was made to concatenate all resulting metrics CSVs from the directory structure the output folder is currently in, into one big CSV.

Current output directory structure:
    output/
    └── <image_name>/
        └── metrics/
            └── <image_name>_metrics_all.csv

Run this from the project root:
    python concat_metrics.py --output ./output --dest ./all_metrics.csv
    python concat_metrics.py --output ./output --dest ./all_metrics.csv --verbose
    
    or alternatively, to get a summary:
    python concat_metrics.py --output ./output --dest ./all_metrics.csv --verbose --summary
"""

import argparse
import pandas as pd
from pathlib import Path


def find_csvs(output_dir: str) -> list[tuple[str, Path]]:
    """
    This function goes through the output directory and finds all metrics CSVs.
    It returns a list of (image_name, csv_path) tuples.
    """
    
    base   = Path(output_dir)
    found  = []

    for image_dir in sorted(base.iterdir()):
        if not image_dir.is_dir():
            continue
        image_name  = image_dir.name
        metrics_dir = image_dir / "metrics"
        if not metrics_dir.exists():
            continue

        # look for any *_metrics_all.csv or metrics_all.csv
        csvs = list(metrics_dir.glob("*metrics_all.csv"))
        if not csvs:
            # fallback solution: any CSV in the metrics dir
            csvs = list(metrics_dir.glob("*.csv"))
        for csv_path in sorted(csvs):
            found.append((image_name, csv_path))

    return found


def concat_metrics(output_dir: str, dest: str, verbose: bool = False) -> pd.DataFrame:
    """
    This function finds all metrics CSVs, adds an Image name column, and concatenates into one DataFrame.
    It saves to destination (dest) and returns the combined DataFrame.
    """
    
    csvs = find_csvs(output_dir)

    if not csvs:
        print(f"No metrics CSVs found under {output_dir}")
        print("Expected structure: output/<image>/metrics/<image>_metrics_all.csv")
        return pd.DataFrame()

    frames = []
    for image_name, csv_path in csvs:
        try:
            df = pd.read_csv(csv_path)
            # insert Image as first column if not already present
            if "Image" not in df.columns:
                df.insert(0, "Image", image_name)
            else:
                df["Image"] = image_name  # overwrite with folder name to be safe
            frames.append(df)
            if verbose:
                print(f"  ✓ {image_name:30s}  {len(df)} rows  ({csv_path})")
        except Exception as e:
            print(f"  ✗ Could not read {csv_path}: {e}")

    if not frames:
        print("No valid CSVs could be read.")
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)

    # reorder columns to the order: Image - standard metadata - metrics
    meta_cols    = ["Image", "Algorithm", "Strategy", "Colour space"]
    metric_cols  = [c for c in combined.columns if c not in meta_cols]
    combined     = combined[[c for c in meta_cols if c in combined.columns] + metric_cols]

    Path(dest).parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(dest, index=False)

    print(f"\nCombined {len(frames)} file(s) → {len(combined)} rows total")
    print(f"   Saved to: {dest}")
    print(f"   Columns:  {list(combined.columns)}")

    return combined


def summarise(df: pd.DataFrame) -> None:
    """As a sanity check let's print a quick summary of the combined DataFrame."""
    
    print("\n── Summary ──────────────────────────────────────────")
    print(f"  Images:        {sorted(df['Image'].unique().tolist())}")
    print(f"  Algorithms:    {sorted(df['Algorithm'].unique().tolist())}")
    print(f"  Strategies:    {sorted(df['Strategy'].unique().tolist())}")
    print(f"  Colour spaces: {sorted(df['Colour space'].unique().tolist())}")

    metric_cols = [c for c in df.columns
                   if c not in ("Image", "Algorithm", "Strategy", "Colour space")]
    if metric_cols:
        print(f"\n── Mean scores per image (across all strategies/cs) ──")
        print(df.groupby("Image")[metric_cols].mean().round(4).to_string())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Concatenate all metrics CSVs.")
    parser.add_argument("--output",  default="./output", help="Root output directory to scan (default: ./output)")
    parser.add_argument("--dest",    default="./all_metrics.csv", help="Destination CSV (default: ./all_metrics.csv)")
    parser.add_argument("--verbose", action="store_true", help="Print each file found")
    parser.add_argument("--summary", action="store_true", help="Print a summary table after concatenating")
    args = parser.parse_args()

    df = concat_metrics(args.output, args.dest, verbose=args.verbose)
    
    if args.summary and not df.empty:
        summarise(df)