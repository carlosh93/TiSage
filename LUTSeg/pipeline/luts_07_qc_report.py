#!/usr/bin/env python3
"""
Optional Step 7: basic QC report for final consensus masks.

Outputs:
- data/LUTS/Annotations/processed/qc_report.json
"""
from __future__ import annotations

import argparse
from collections import Counter

import cv2
import numpy as np

from common import VALID_MASK_VALUES, parse_json, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate LUTS mask QC report.")
    parser.add_argument(
        "--consensus-manifest",
        default="data/LUTS/Annotations/processed/consensus_manifest.json",
        help="Consensus manifest JSON path.",
    )
    parser.add_argument(
        "--output-json",
        default="data/LUTS/Annotations/processed/qc_report.json",
        help="Output QC report path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = parse_json(args.consensus_manifest)
    if not isinstance(manifest, list) or not manifest:
        raise SystemExit("Consensus manifest is empty or invalid.")

    total_pixels_by_class: Counter[int] = Counter()
    per_image = []
    issues = []

    for item in manifest:
        mask_path = item.get("mask_path")
        image_key = item.get("image_key")
        if not mask_path:
            continue
        mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
        if mask is None:
            issues.append({"image_key": image_key, "issue": "mask_missing"})
            continue
        if mask.ndim != 2:
            issues.append({"image_key": image_key, "issue": "mask_not_single_channel"})
            continue

        values, counts = np.unique(mask, return_counts=True)
        values_set = set(values.tolist())
        if not values_set.issubset(VALID_MASK_VALUES):
            issues.append(
                {
                    "image_key": image_key,
                    "issue": "unexpected_values",
                    "values": sorted(values_set),
                }
            )

        class_counts = {int(v): int(c) for v, c in zip(values, counts)}
        for v, c in class_counts.items():
            total_pixels_by_class[v] += c

        total = int(mask.size)
        ignore_ratio = class_counts.get(255, 0) / total
        per_image.append(
            {
                "image_key": image_key,
                "patient_id": item.get("patient_id"),
                "mask_path": mask_path,
                "n_annotators": item.get("n_annotators"),
                "ignore_ratio": round(ignore_ratio, 6),
                "class_pixel_counts": class_counts,
            }
        )

    report = {
        "n_images": len(per_image),
        "class_pixel_totals": {str(k): int(v) for k, v in sorted(total_pixels_by_class.items())},
        "issues": issues,
        "images": per_image,
    }
    write_json(args.output_json, report)
    print(f"Wrote QC report: {args.output_json}")
    print(f"Images processed: {report['n_images']}")
    print(f"Issues found: {len(issues)}")


if __name__ == "__main__":
    main()
