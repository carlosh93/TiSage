#!/usr/bin/env python3
"""
Step 1: Normalize raw Label Studio exports for LUTS.

Inputs:
- data/LUTS/Annotations/raw/*.json

Outputs:
- data/LUTS/Annotations/processed/normalized_annotations.json
- data/LUTS/Annotations/processed/normalize_summary.json
"""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from common import (
    CLASS_ID_TO_NAME,
    SummaryCounter,
    WOUND_LABEL_ALIASES,
    canonical_image_key,
    image_relpath_from_value,
    infer_patient_id,
    normalize_label,
    parse_json,
    tissue_label_to_id,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Normalize LUTS Label Studio exports.")
    parser.add_argument(
        "--raw-dir",
        default="data/LUTS/Annotations/raw",
        help="Directory containing raw Label Studio export JSON files.",
    )
    parser.add_argument(
        "--output-json",
        default="data/LUTS/Annotations/processed/normalized_annotations.json",
        help="Output JSON path for normalized records.",
    )
    parser.add_argument(
        "--summary-json",
        default="data/LUTS/Annotations/processed/normalize_summary.json",
        help="Output JSON path for summary statistics.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw_dir = Path(args.raw_dir).resolve()
    raw_files = sorted(raw_dir.glob("*.json"))
    if not raw_files:
        raise SystemExit(f"No raw exports found in: {raw_dir}")

    records: list[dict] = []
    class_counter: Counter[int] = Counter()
    counter = SummaryCounter()

    for export_path in raw_files:
        raw_tasks = parse_json(export_path)
        if not isinstance(raw_tasks, list):
            continue

        for task in raw_tasks:
            task_id = task.get("id")
            project_id = task.get("project")
            data = task.get("data", {}) or {}
            image_value = str(data.get("image", ""))
            image_name = data.get("image_name")
            patient_id = infer_patient_id(data)
            image_relpath = image_relpath_from_value(image_value)
            image_key = canonical_image_key(image_relpath)
            source_doctor_fallback = "unknown"

            annotations = task.get("annotations", []) or []
            if not annotations:
                counter.skipped_no_annotations += 1
                continue

            for annotation in annotations:
                completed_by = annotation.get("completed_by")
                doctor_id = (
                    f"user_{completed_by}" if completed_by is not None else source_doctor_fallback
                )
                result_items = annotation.get("result", []) or []
                polygons: list[dict] = []
                wound_polygons: list[dict] = []
                for item in result_items:
                    if item.get("type") != "polygonlabels":
                        continue
                    value = item.get("value", {}) or {}
                    labels = value.get("polygonlabels", []) or []
                    points = value.get("points", []) or []
                    if not labels or not points:
                        continue

                    raw_label = labels[0]
                    if normalize_label(raw_label) in WOUND_LABEL_ALIASES:
                        wound_polygons.append(
                            {
                                "raw_label": raw_label,
                                "from_name": item.get("from_name"),
                                "to_name": item.get("to_name"),
                                "original_width": item.get("original_width")
                                or value.get("original_width"),
                                "original_height": item.get("original_height")
                                or value.get("original_height"),
                                "points": points,
                                "closed": value.get("closed", True),
                            }
                        )
                        continue

                    class_id = tissue_label_to_id(raw_label)
                    if class_id is None:
                        # Ignore any unknown control labels.
                        continue

                    polygon = {
                        "class_id": class_id,
                        "class_name": CLASS_ID_TO_NAME[class_id],
                        "raw_label": raw_label,
                        "from_name": item.get("from_name"),
                        "to_name": item.get("to_name"),
                        "original_width": item.get("original_width")
                        or value.get("original_width"),
                        "original_height": item.get("original_height")
                        or value.get("original_height"),
                        "points": points,
                        "closed": value.get("closed", True),
                    }
                    polygons.append(polygon)
                    class_counter[class_id] += 1

                if not polygons:
                    counter.skipped_no_tissue += 1
                    continue

                record = {
                    "export_file": export_path.name,
                    "project_id": project_id,
                    "doctor_id": doctor_id,
                    "task_id": task_id,
                    "annotation_id": annotation.get("id"),
                    "completed_by": completed_by,
                    "patient_id": patient_id,
                    "image_name": image_name,
                    "image_value": image_value,
                    "image_relpath": image_relpath,
                    "image_key": image_key,
                    "polygons": polygons,
                    "wound_polygons": wound_polygons,
                }
                records.append(record)
                counter.records += 1
                counter.polygons_total += len(polygons)

    if not records:
        raise SystemExit("No normalized records were generated.")

    write_json(args.output_json, records)

    summary = {
        "raw_files": [p.name for p in raw_files],
        "records": counter.records,
        "polygons_total": counter.polygons_total,
        "skipped_no_annotations": counter.skipped_no_annotations,
        "skipped_no_tissue": counter.skipped_no_tissue,
        "class_polygon_counts": {str(k): int(v) for k, v in sorted(class_counter.items())},
    }
    write_json(args.summary_json, summary)

    print(f"Wrote normalized records: {args.output_json}")
    print(f"Wrote summary: {args.summary_json}")
    print("Class polygon counts:", summary["class_polygon_counts"])


if __name__ == "__main__":
    main()
