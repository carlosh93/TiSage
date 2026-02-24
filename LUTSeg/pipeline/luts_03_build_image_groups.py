#!/usr/bin/env python3
"""
Step 3: Group per-annotator tissue/wound masks by unique image.

Input:
- data/LUTS/Annotations/processed/rasterized_manifest.json

Output:
- data/LUTS/Annotations/processed/image_groups.json
"""
from __future__ import annotations

import argparse
from collections import defaultdict

from common import parse_csv_arg, parse_json, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Group LUTS masks by image key.")
    parser.add_argument(
        "--manifest-json",
        default="data/LUTS/Annotations/processed/rasterized_manifest.json",
        help="Rasterized manifest path.",
    )
    parser.add_argument(
        "--output-json",
        default="data/LUTS/Annotations/processed/image_groups.json",
        help="Output grouped JSON path.",
    )
    parser.add_argument(
        "--golden-patients",
        default="Patient_1,Patient_5,Patient_6,Patient_11,Patient_13,Patient_20,Patient_28,Patient_33,Patient_39",
        help="Comma-separated golden patient IDs (for metadata tagging).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    golden_set = set(parse_csv_arg(args.golden_patients))
    manifest = parse_json(args.manifest_json)
    if not isinstance(manifest, list) or not manifest:
        raise SystemExit("Rasterized manifest is empty or invalid.")

    grouped: dict[str, list[dict]] = defaultdict(list)
    for item in manifest:
        key = item.get("image_key") or item.get("image_name")
        if not key:
            continue
        grouped[key].append(item)

    groups: list[dict] = []
    for key in sorted(grouped.keys()):
        items = grouped[key]
        first = items[0]
        patient_id = first.get("patient_id")
        doctors = sorted({str(i.get("doctor_id")) for i in items})
        groups.append(
            {
                "image_key": key,
                "image_name": first.get("image_name"),
                "patient_id": patient_id,
                "is_golden_patient": patient_id in golden_set if patient_id else False,
                "n_annotators": len(items),
                "doctor_ids": doctors,
                "source_image_path": first.get("image_path"),
                "annotations": [
                    {
                        "doctor_id": i.get("doctor_id"),
                        "annotation_id": i.get("annotation_id"),
                        "mask_path": i.get("mask_path"),
                        "wound_mask_path": i.get("wound_mask_path"),
                        "project_id": i.get("project_id"),
                        "task_id": i.get("task_id"),
                    }
                    for i in items
                ],
            }
        )

    write_json(args.output_json, groups)
    print(f"Grouped unique images: {len(groups)}")
    print(f"Wrote: {args.output_json}")


if __name__ == "__main__":
    main()
