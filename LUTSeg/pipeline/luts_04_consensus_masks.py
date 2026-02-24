#!/usr/bin/env python3
"""
Step 4: Build one final mask per image.

Rules:
- If only one annotation exists for an image -> use it directly.
- If multiple annotations exist -> pixelwise majority vote over valid labels (0..5),
  ignoring 255.
- Tie policy is configurable (default: 255 ignore).

Outputs:
- data/LUTS/Masks/**.png
- data/LUTS/Wound_Masks/**.png
- data/LUTS/Annotations/processed/consensus_manifest.json
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from common import IGNORE_VALUE, VALID_MASK_VALUES, parse_json, write_json


def smooth_mask(mask: np.ndarray, kernel_size: int) -> np.ndarray:
    """
    Smooth mask and fill small gaps using a mode (majority) filter.
    - Fills holes and gaps where a single class dominates in the neighborhood.
    - Smooths jagged boundaries.
    Pixels with value 255 (ignore) are not used when computing the mode;
    they get replaced by the majority of surrounding valid labels (0..5).
    """
    if kernel_size <= 1:
        return mask
    h, w = mask.shape
    r = kernel_size // 2  # e.g. 5 -> 2
    pad = r
    padded = np.full((h + 2 * pad, w + 2 * pad), IGNORE_VALUE, dtype=np.uint8)
    padded[pad : pad + h, pad : pad + w] = mask

    # Build stack of shifted views: (k*k, H, W)
    views = []
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            views.append(padded[pad + dy : pad + dy + h, pad + dx : pad + dx + w])
    stack = np.stack(views, axis=0)

    # For each pixel: majority among values in {0..5} (exclude 255)
    counts = np.zeros((6, h, w), dtype=np.uint16)
    for cls in range(6):
        counts[cls] = (stack == cls).sum(axis=0)
    total_valid = counts.sum(axis=0)
    best_cls = np.argmax(counts, axis=0).astype(np.uint8)
    out = np.where(total_valid > 0, best_cls, IGNORE_VALUE).astype(np.uint8)
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create consensus LUTS masks.")
    parser.add_argument(
        "--groups-json",
        default="data/LUTS/Annotations/processed/image_groups.json",
        help="Image groups JSON path.",
    )
    parser.add_argument(
        "--output-dir",
        default="data/LUTS/Masks",
        help="Output directory for final consensus masks.",
    )
    parser.add_argument(
        "--wound-output-dir",
        default="data/LUTS/Wound_Masks",
        help="Output directory for final wound masks.",
    )
    parser.add_argument(
        "--manifest-json",
        default="data/LUTS/Annotations/processed/consensus_manifest.json",
        help="Output consensus manifest path.",
    )
    parser.add_argument(
        "--tie-policy",
        choices=["ignore", "lowest"],
        default="ignore",
        help="How to resolve ties in multi-annotator vote.",
    )
    parser.add_argument(
        "--min-golden-annotators",
        type=int,
        default=1,
        help=(
            "For golden patients, skip consensus if available annotators are below this "
            "number. Use 4 now, then 5 when D5 is complete."
        ),
    )
    parser.add_argument(
        "--smooth-kernel",
        type=int,
        default=5,
        metavar="N",
        help=(
            "Post-process mask with a mode (majority) filter of size N×N to smooth "
            "boundaries and fill small gaps. Use 0 to disable. Default: 5."
        ),
    )
    return parser.parse_args()


def majority_vote(stack: np.ndarray, tie_policy: str) -> np.ndarray:
    """
    stack: (N, H, W), uint8 masks with values in {0..5, 255}
    """
    n, h, w = stack.shape
    out = np.full((h, w), IGNORE_VALUE, dtype=np.uint8)

    valid = stack != IGNORE_VALUE
    any_valid = valid.any(axis=0)
    out[~any_valid] = IGNORE_VALUE

    counts = np.zeros((6, h, w), dtype=np.uint16)  # classes 0..5
    for cls in range(6):
        counts[cls] = np.sum((stack == cls) & valid, axis=0)

    best_cls = np.argmax(counts, axis=0).astype(np.uint8)
    best_count = np.max(counts, axis=0)

    tie_mask = np.zeros((h, w), dtype=bool)
    for cls in range(6):
        tie_mask |= (counts[cls] == best_count) & (best_count > 0)
    # tie_mask currently marks all max positions, including unique maxima.
    # Count how many classes match best_count to detect true ties.
    n_max = np.zeros((h, w), dtype=np.uint8)
    for cls in range(6):
        n_max += ((counts[cls] == best_count) & (best_count > 0)).astype(np.uint8)
    true_tie = n_max > 1

    out[any_valid] = best_cls[any_valid]
    if tie_policy == "ignore":
        out[true_tie] = IGNORE_VALUE

    return out


def majority_vote_binary(stack: np.ndarray) -> np.ndarray:
    """
    stack: (N, H, W), uint8 wound masks in {0, 255}
    Returns uint8 mask in {0, 255} with pixel-wise majority.
    """
    n = stack.shape[0]
    positive = (stack > 0).astype(np.uint8)
    votes = positive.sum(axis=0)
    threshold = (n + 1) // 2  # ceil(n/2)
    return np.where(votes >= threshold, 255, 0).astype(np.uint8)


def main() -> None:
    args = parse_args()
    groups = parse_json(args.groups_json)
    if not isinstance(groups, list) or not groups:
        raise SystemExit("Image groups JSON is empty or invalid.")

    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    wound_out_dir = Path(args.wound_output_dir).resolve()
    wound_out_dir.mkdir(parents=True, exist_ok=True)

    manifest: list[dict] = []
    skipped_missing = 0
    skipped_golden_min = 0

    for group in groups:
        patient_id = group.get("patient_id") or "unknown_patient"
        image_name = group.get("image_name")
        image_key = group.get("image_key")
        annotations = group.get("annotations", [])
        is_golden = bool(group.get("is_golden_patient"))

        if not image_name or not annotations:
            continue

        if is_golden and len(annotations) < args.min_golden_annotators:
            skipped_golden_min += 1
            continue

        masks: list[np.ndarray] = []
        wound_masks: list[np.ndarray] = []
        for ann in annotations:
            mask_path = ann.get("mask_path")
            if not mask_path:
                continue
            arr = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
            if arr is None:
                continue
            if arr.ndim != 2:
                raise SystemExit(f"Mask is not single-channel: {mask_path}")
            unique = set(np.unique(arr).tolist())
            if not unique.issubset(VALID_MASK_VALUES):
                raise SystemExit(
                    f"Unexpected values {sorted(unique)} in mask: {mask_path}"
                )
            masks.append(arr.astype(np.uint8))

            wound_path = ann.get("wound_mask_path")
            if wound_path:
                wound_arr = cv2.imread(wound_path, cv2.IMREAD_UNCHANGED)
                if wound_arr is not None:
                    if wound_arr.ndim != 2:
                        raise SystemExit(f"Wound mask is not single-channel: {wound_path}")
                    wound_bin = np.where(wound_arr > 0, 255, 0).astype(np.uint8)
                    # Ignore empty wound masks (e.g. annotation had no wound_outline).
                    if np.any(wound_bin > 0):
                        wound_masks.append(wound_bin)

        if not masks:
            skipped_missing += 1
            continue

        if len(masks) == 1:
            consensus = masks[0]
            method = "single_annotator"
        else:
            stack = np.stack(masks, axis=0)
            consensus = majority_vote(stack, tie_policy=args.tie_policy)
            method = "majority_vote"

        # Wound consensus from wound_outline masks if present; fallback to tissue extent.
        if wound_masks:
            if len(wound_masks) == 1:
                wound_consensus = wound_masks[0]
                wound_method = "single_annotator_wound"
            else:
                wound_consensus = majority_vote_binary(np.stack(wound_masks, axis=0))
                wound_method = "majority_vote_wound"
        else:
            wound_consensus = np.where(consensus > 0, 255, 0).astype(np.uint8)
            wound_method = "derived_from_tissue_mask"

        if args.smooth_kernel > 0:
            consensus = smooth_mask(consensus, args.smooth_kernel)

        patient_dir = out_dir / str(patient_id)
        patient_dir.mkdir(parents=True, exist_ok=True)
        out_path = patient_dir / f"{Path(image_name).stem}.png"
        cv2.imwrite(str(out_path), consensus)

        wound_patient_dir = wound_out_dir / str(patient_id)
        wound_patient_dir.mkdir(parents=True, exist_ok=True)
        wound_out_path = wound_patient_dir / f"{Path(image_name).stem}.png"
        cv2.imwrite(str(wound_out_path), wound_consensus)

        manifest.append(
            {
                "image_key": image_key,
                "image_name": image_name,
                "patient_id": patient_id,
                "is_golden_patient": is_golden,
                "n_annotators": len(masks),
                "doctor_ids": sorted({str(a.get("doctor_id")) for a in annotations}),
                "method": method,
                "tie_policy": args.tie_policy,
                "source_image_path": group.get("source_image_path"),
                "mask_path": str(out_path),
                "wound_mask_path": str(wound_out_path),
                "wound_method": wound_method,
            }
        )

    if not manifest:
        raise SystemExit("No consensus masks were generated.")

    write_json(args.manifest_json, manifest)
    print(f"Consensus masks generated: {len(manifest)}")
    print(f"Skipped (golden below min annotators): {skipped_golden_min}")
    print(f"Skipped (missing masks): {skipped_missing}")
    print(f"Wrote manifest: {args.manifest_json}")


if __name__ == "__main__":
    main()
