"""Generate ground truth pairwise distance matrices for datasets."""

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
TEXTS_DIR = ROOT / "data" / "texts"
CONSTANTS_DIR = ROOT / "data" / "constants"
GRAM_DIR = ROOT / "data" / "gram"


def generate_absolute(name: str):
    """Generate |a - b| distance matrix for a dataset of integer labels."""
    labels = (TEXTS_DIR / f"{name}.txt").read_text().strip().splitlines()
    values_B = np.array([int(v) for v in labels])

    abs_diff_BB = np.abs(values_B[:, None] - values_B[None, :])
    df = pd.DataFrame(abs_diff_BB, index=labels, columns=labels)

    out_path = GRAM_DIR / f"{name}_absolute.csv"
    df.to_csv(out_path)
    print(f"Wrote {out_path}  ({len(labels)}x{len(labels)}, max={abs_diff_BB.max()})")


def generate_log_ratio(name: str):
    """Generate |log(a / b)| distance matrix for a dataset of integer labels."""
    labels = (TEXTS_DIR / f"{name}.txt").read_text().strip().splitlines()
    values_B = np.array([int(v) for v in labels])

    if np.any(values_B <= 0):
        print(f"Skipping log_ratio for {name} (contains non-positive values)")
        return

    log_ratio_BB = np.abs(np.log(values_B[:, None] / values_B[None, :]))
    df = pd.DataFrame(log_ratio_BB, index=labels, columns=labels)

    out_path = GRAM_DIR / f"{name}_log_ratio.csv"
    df.to_csv(out_path)
    print(f"Wrote {out_path}  ({len(labels)}x{len(labels)})")


def generate_circular(name: str):
    """Generate circular distance matrix for an ordered cyclic dataset.

    Elements are assumed to be equally spaced around a cycle in file order.
    Distance is min(|i-j|, N-|i-j|) where N is the number of elements.
    """
    labels = (TEXTS_DIR / f"{name}.txt").read_text().strip().splitlines()
    n = len(labels)
    idx_B = np.arange(n)
    raw_diff_BB = np.abs(idx_B[:, None] - idx_B[None, :])
    circ_dist_BB = np.minimum(raw_diff_BB, n - raw_diff_BB)

    df = pd.DataFrame(circ_dist_BB, index=labels, columns=labels)
    out_path = GRAM_DIR / f"{name}_circular.csv"
    df.to_csv(out_path)
    print(f"Wrote {out_path}  ({n}x{n}, max={circ_dist_BB.max()})")


def generate_hue_distance(name: str):
    """Generate circular hue distance matrix for a color LCH dataset.

    Uses min(|h1-h2|, 360-|h1-h2|) as the distance on the hue circle.
    Color names are looked up in data/constants/color_conversion.csv (column H).
    """
    labels = (TEXTS_DIR / f"{name}.txt").read_text().strip().splitlines()
    # Labels have " color" suffix, strip it for CSV lookup
    bare_names = [label.removesuffix(" color") for label in labels]

    conversion = pd.read_csv(CONSTANTS_DIR / "color_conversion.csv")
    name_to_hue = dict(zip(conversion["color_name"], conversion["H"]))

    missing = [n for n in bare_names if n not in name_to_hue]
    if missing:
        raise ValueError(f"Colors not found in conversion CSV: {missing}")

    hue_B = np.array([name_to_hue[n] for n in bare_names])
    raw_diff_BB = np.abs(hue_B[:, None] - hue_B[None, :])
    hue_dist_BB = np.minimum(raw_diff_BB, 360.0 - raw_diff_BB)

    df = pd.DataFrame(hue_dist_BB, index=labels, columns=labels)
    out_path = GRAM_DIR / f"{name}_hue.csv"
    df.to_csv(out_path)
    print(f"Wrote {out_path}  ({len(labels)}x{len(labels)}, max={hue_dist_BB.max():.1f})")


if __name__ == "__main__":
    GRAM_DIR.mkdir(parents=True, exist_ok=True)

    integer_datasets = ["integers100", "integers500", "integers1000", "integers1000step5"]
    for name in integer_datasets:
        generate_absolute(name)

    year_datasets = ["years100"]
    for name in year_datasets:
        generate_absolute(name)
        generate_log_ratio(name)

    color_lch_datasets = ["colors101_lch", "colors195_lch"]
    for name in color_lch_datasets:
        generate_hue_distance(name)

    cyclic_datasets = ["days7", "months12"]
    for name in cyclic_datasets:
        generate_circular(name)
