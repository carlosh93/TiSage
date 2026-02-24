#!/usr/bin/env python3
"""
Inter-rater variability: Figure 1 (ICC by tissue + Dice distribution).

Uses the golden set from image_groups.json. Computes:
- Tissue proportions per (image, doctor) from rasterized masks.
- ICC(3,1) per tissue type (Pingouin).
- Pairwise Dice per image (tissue vs background) across annotators.

Outputs:
- Figure 1: (A) ICC bar plot by tissue with 95% CI, (B) Dice distribution.
- Optional CSVs: proportions, ICC table, Dice per image.

Run from repo root (use project venv if you have one):
  venv/bin/python data/LUTS/pipeline/luts_inter_rater_figure1.py [--save-csvs]

Requires: numpy, pandas, opencv-python, matplotlib, pingouin (pip install pingouin)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

# Pipeline common (CLASS_ID_TO_NAME, IGNORE_VALUE)
REPO_ROOT = Path(__file__).resolve().parents[3]
PIPELINE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PIPELINE_DIR))
from common import CLASS_ID_TO_NAME, IGNORE_VALUE, parse_json

try:
    import pingouin as pg
except ImportError:
    raise SystemExit("Install pingouin: pip install pingouin")

import matplotlib.pyplot as plt

# Tissue class IDs (1..5); 0 = background, 255 = ignore
TISSUE_IDS = sorted(k for k in CLASS_ID_TO_NAME if k != 0)
TISSUE_NAMES = [CLASS_ID_TO_NAME[i] for i in TISSUE_IDS]
# Short names for plot
TISSUE_SHORT = {
    1: "Epithelial",
    2: "Slough",
    3: "Granulation",
    4: "Necrotic",
    5: "Other",
}


def compute_proportions(mask: np.ndarray) -> dict[int, float]:
    """Proportions of tissue classes 1..5 as fraction of non-background, non-ignore pixels."""
    valid = (mask >= 1) & (mask <= 5)
    total = int(valid.sum())
    if total == 0:
        return {i: 0.0 for i in TISSUE_IDS}
    out = {}
    for cid in TISSUE_IDS:
        out[cid] = float((mask == cid).sum()) / total
    return out


def load_mask(path: str | Path) -> np.ndarray | None:
    arr = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if arr is None:
        return None
    if arr.ndim != 2:
        return None
    return arr.astype(np.uint8)


def dice_binary(m1: np.ndarray, m2: np.ndarray) -> float:
    """Dice for binary masks (same shape). Expects 0/1 or 0/255."""
    a = (m1 > 0).astype(np.uint8)
    b = (m2 > 0).astype(np.uint8)
    inter = (a & b).sum()
    total = a.sum() + b.sum()
    if total == 0:
        return 1.0
    return 2.0 * inter / total


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inter-rater Figure 1: ICC + Dice.")
    parser.add_argument(
        "--groups-json",
        default="data/LUTS/Annotations/processed/image_groups.json",
        help="Image groups JSON (must contain is_golden_patient).",
    )
    parser.add_argument(
        "--golden-only",
        action="store_true",
        default=True,
        help="Restrict to golden set (default: True).",
    )
    parser.add_argument(
        "--output-figure",
        default="data/LUTS/Annotations/processed/inter_rater_figure1.png",
        help="Output path for Figure 1.",
    )
    parser.add_argument(
        "--output-dir",
        default="data/LUTS/Annotations/processed",
        help="Directory for optional CSV outputs.",
    )
    parser.add_argument(
        "--save-csvs",
        action="store_true",
        help="Save proportions, ICC, and Dice tables as CSV.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=150,
        help="Figure DPI.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(REPO_ROOT)
    groups_path = root / args.groups_json
    groups = parse_json(groups_path)
    if not isinstance(groups, list):
        raise SystemExit("groups_json must be a list of groups.")

    if args.golden_only:
        groups = [g for g in groups if g.get("is_golden_patient")]
    if not groups:
        raise SystemExit("No groups to analyze (empty or no golden).")

    # --- Build proportions table (long format) ---
    rows = []
    for g in groups:
        image_key = g.get("image_key")
        if not image_key:
            continue
        for ann in g.get("annotations", []):
            doctor_id = ann.get("doctor_id")
            mask_path = ann.get("mask_path")
            if not doctor_id or not mask_path:
                continue
            full_path = root / mask_path if not Path(mask_path).is_absolute() else Path(mask_path)
            if not full_path.exists():
                continue
            mask = load_mask(full_path)
            if mask is None:
                continue
            prop = compute_proportions(mask)
            row = {"image_key": image_key, "doctor_id": doctor_id}
            for cid in TISSUE_IDS:
                row[TISSUE_SHORT[cid]] = prop[cid]
            rows.append(row)

    df = pd.DataFrame(rows)
    if df.empty:
        raise SystemExit("No valid masks found.")

    # --- ICC per tissue (Pingouin) ---
    icc_results = []
    for cid in TISSUE_IDS:
        col = TISSUE_SHORT[cid]
        try:
            icc = pg.intraclass_corr(
                data=df,
                targets="image_key",
                raters="doctor_id",
                ratings=col,
            )
            # Use ICC3 (single fixed raters) to match JMIR-style reporting
            row_icc3 = icc[icc["Type"] == "ICC3"].iloc[0]
            icc_results.append({
                "tissue": col,
                "ICC": row_icc3["ICC"],
                "CI95%_lo": row_icc3["CI95%"][0],
                "CI95%_hi": row_icc3["CI95%"][1],
            })
        except Exception as e:
            icc_results.append({"tissue": col, "ICC": np.nan, "CI95%_lo": np.nan, "CI95%_hi": np.nan})

    icc_df = pd.DataFrame(icc_results)

    # --- Dice: pairwise per image ---
    dice_list = []
    for g in groups:
        image_key = g.get("image_key")
        anns = g.get("annotations", [])
        if len(anns) < 2:
            continue
        masks = []
        for ann in anns:
            mask_path = ann.get("mask_path")
            if not mask_path:
                continue
            full_path = root / mask_path if not Path(mask_path).is_absolute() else Path(mask_path)
            if not full_path.exists():
                continue
            m = load_mask(full_path)
            if m is None:
                continue
            # Binary: tissue (1..5) vs background (0, 255)
            binary = ((m >= 1) & (m <= 5)).astype(np.uint8)
            masks.append((ann.get("doctor_id"), binary))

        if len(masks) < 2:
            continue
        # Resize all to same shape (first mask)
        h, w = masks[0][1].shape
        for i in range(len(masks)):
            if masks[i][1].shape != (h, w):
                masks[i] = (masks[i][0], cv2.resize(
                    masks[i][1], (w, h), interpolation=cv2.INTER_NEAREST
                ))
        # All pairs
        for i in range(len(masks)):
            for j in range(i + 1, len(masks)):
                d = dice_binary(masks[i][1], masks[j][1])
                dice_list.append({
                    "image_key": image_key,
                    "rater_a": masks[i][0],
                    "rater_b": masks[j][0],
                    "dice": d,
                })

    dice_df = pd.DataFrame(dice_list)
    if dice_df.empty:
        dice_values = np.array([np.nan])
    else:
        dice_values = dice_df["dice"].values

    # --- Figure 1: (A) ICC by tissue, (B) Dice distribution ---
    fig, (ax_icc, ax_dice) = plt.subplots(1, 2, figsize=(10, 4))

    # (A) ICC
    x = np.arange(len(icc_df))
    bars = ax_icc.bar(
        x - 0.2,
        icc_df["ICC"],
        width=0.4,
        yerr=[
            icc_df["ICC"] - icc_df["CI95%_lo"],
            icc_df["CI95%_hi"] - icc_df["ICC"],
        ],
        capsize=4,
        color="steelblue",
        edgecolor="black",
        linewidth=0.8,
    )
    ax_icc.set_xticks(x)
    ax_icc.set_xticklabels(icc_df["tissue"], rotation=15, ha="right")
    ax_icc.set_ylabel("ICC(3,1)")
    ax_icc.set_title("Inter-rater agreement (tissue proportions)")
    ax_icc.axhline(0.5, color="gray", linestyle="--", linewidth=0.8)
    ax_icc.set_ylim(0, 1.05)
    ax_icc.set_xlabel("Tissue type")

    # (B) Dice
    ax_dice.boxplot(
        dice_values,
        vert=True,
        patch_artist=True,
        widths=0.5,
        showfliers=True,
    )
    for patch in ax_dice.patches:
        patch.set_facecolor("lightsteelblue")
        patch.set_edgecolor("black")
    ax_dice.set_ylabel("Dice coefficient")
    ax_dice.set_title("Pairwise segmentation overlap (tissue vs background)")
    ax_dice.set_xticklabels(["All pairs"])
    ax_dice.set_ylim(0, 1.05)
    # Add n and mean as text
    n_pairs = len(dice_values)
    mean_d = np.nanmean(dice_values)
    ax_dice.text(0.98, 0.02, f"n = {n_pairs} pairs\nmean = {mean_d:.3f}", transform=ax_dice.transAxes, fontsize=9, va="bottom", ha="right")

    plt.tight_layout()
    out_fig = root / args.output_figure
    out_fig.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(out_fig), dpi=args.dpi, bbox_inches="tight")
    plt.close()
    print(f"Figure 1 saved: {out_fig}")

    if args.save_csvs:
        out_dir = root / args.output_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_dir / "inter_rater_proportions.csv", index=False)
        icc_df.to_csv(out_dir / "inter_rater_ICC.csv", index=False)
        dice_df.to_csv(out_dir / "inter_rater_dice_pairs.csv", index=False)
        print(f"CSVs saved in {out_dir}")


if __name__ == "__main__":
    main()
