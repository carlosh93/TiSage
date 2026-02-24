#!/usr/bin/env python3
"""
Step 2: Rasterize normalized polygons into per-annotation mask PNGs.
If wound_outline exists: fill holes inside the wound, and set all pixels
outside the wound boundary to background (0), clipping any tissue drawn outside.

Input:
- data/LUTS/Annotations/processed/normalized_annotations.json

Output:
- data/LUTS/Annotations/processed/masks_by_annotator/**.png
- data/LUTS/Annotations/processed/wound_masks_by_annotator/**.png
- data/LUTS/Annotations/processed/rasterized_manifest.json
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from common import (
    VALID_MASK_VALUES,
    build_basename_cache,
    find_image_path,
    parse_json,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rasterize LUTS polygon annotations.")
    parser.add_argument(
        "--normalized-json",
        default="data/LUTS/Annotations/processed/normalized_annotations.json",
        help="Path to normalized annotations JSON.",
    )
    parser.add_argument(
        "--images-root",
        default="data/Dataset_evolution_wounds_VR",
        help="Root directory containing source images.",
    )
    parser.add_argument(
        "--output-dir",
        default="data/LUTS/Annotations/processed/masks_by_annotator",
        help="Output directory for per-annotator masks.",
    )
    parser.add_argument(
        "--wound-output-dir",
        default="data/LUTS/Annotations/processed/wound_masks_by_annotator",
        help="Output directory for per-annotator wound masks (from wound_outline).",
    )
    parser.add_argument(
        "--manifest-json",
        default="data/LUTS/Annotations/processed/rasterized_manifest.json",
        help="Output manifest JSON path.",
    )
    return parser.parse_args()


def polygon_percent_to_pixels(points: list[list[float]], w: int, h: int) -> np.ndarray:
    pixel_points: list[list[int]] = []
    for px, py in points:
        x = int(round((float(px) / 100.0) * max(w - 1, 1)))
        y = int(round((float(py) / 100.0) * max(h - 1, 1)))
        pixel_points.append([x, y])
    return np.array(pixel_points, dtype=np.int32)


def fill_unlabeled_inside_wound(mask: np.ndarray, wound_mask: np.ndarray) -> tuple[np.ndarray, int]:
    """
    Fill unlabeled pixels (value 0) inside wound outline using nearest tissue class (1..5).
    Returns (filled_mask, n_filled_pixels).
    """
    holes = (wound_mask > 0) & (mask == 0)
    n_holes = int(np.count_nonzero(holes))
    if n_holes == 0:
        return mask, 0

    present_classes = [cls for cls in range(1, 6) if np.any(mask == cls)]
    if not present_classes:
        return mask, 0

    dists = []
    for cls in present_classes:
        # Distance to nearest pixel of this class.
        # distanceTransform computes distance to nearest zero pixel, so class pixels are 0.
        src = np.where(mask == cls, 0, 255).astype(np.uint8)
        dists.append(cv2.distanceTransform(src, cv2.DIST_L2, 3))
    dist_stack = np.stack(dists, axis=0)  # (C, H, W)
    nearest_idx = np.argmin(dist_stack, axis=0)
    class_lut = np.array(present_classes, dtype=np.uint8)
    nearest_class = class_lut[nearest_idx]

    out = mask.copy()
    out[holes] = nearest_class[holes]
    return out, n_holes


def main() -> None:
    args = parse_args()
    records = parse_json(args.normalized_json)
    if not isinstance(records, list) or not records:
        raise SystemExit("Normalized JSON is empty or invalid.")

    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    wound_out_dir = Path(args.wound_output_dir).resolve()
    wound_out_dir.mkdir(parents=True, exist_ok=True)

    basename_cache = build_basename_cache(args.images_root)

    manifest: list[dict] = []
    skipped_missing_image = 0
    total_filled_inside_wound = 0

    for rec in records:
        image_name = rec.get("image_name")
        image_key = rec.get("image_key", "")
        image_relpath = rec.get("image_relpath", "")
        patient_id = rec.get("patient_id") or "unknown_patient"
        doctor_id = rec.get("doctor_id") or "unknown_doctor"
        annotation_id = rec.get("annotation_id") or "unknown_ann"
        polygons = rec.get("polygons", [])
        wound_polygons = rec.get("wound_polygons", [])

        image_path = find_image_path(
            image_key=image_relpath,
            image_name=image_name,
            images_root=args.images_root,
            basename_cache=basename_cache,
        )
        if not image_path:
            skipped_missing_image += 1
            continue

        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if image is None:
            skipped_missing_image += 1
            continue
        h, w = image.shape[:2]

        mask = np.zeros((h, w), dtype=np.uint8)
        raster_items: list[tuple[float, int, np.ndarray]] = []
        for poly in polygons:
            class_id = int(poly["class_id"])
            points = poly.get("points", [])
            if not points:
                continue
            contour = polygon_percent_to_pixels(points, w, h)
            if contour.shape[0] < 3:
                continue
            area = float(cv2.contourArea(contour))
            raster_items.append((area, class_id, contour))

        # Draw larger polygons first, smaller ones last.
        # This preserves "small regions inside large regions" regardless of export order.
        for _, class_id, contour in sorted(raster_items, key=lambda t: t[0], reverse=True):
            cv2.fillPoly(mask, [contour], color=class_id)

        wound_mask = np.zeros((h, w), dtype=np.uint8)
        if wound_polygons:
            for poly in wound_polygons:
                points = poly.get("points", [])
                if not points:
                    continue
                contour = polygon_percent_to_pixels(points, w, h)
                if contour.shape[0] < 3:
                    continue
                cv2.fillPoly(wound_mask, [contour], color=1)
            mask, n_filled = fill_unlabeled_inside_wound(mask, wound_mask)
            total_filled_inside_wound += n_filled
            # Clip to wound: discard any pixel outside the boundary (e.g. tissue drawn slightly outside)
            mask[wound_mask == 0] = 0

        unique_vals = set(np.unique(mask).tolist())
        if not unique_vals.issubset(VALID_MASK_VALUES):
            raise SystemExit(
                f"Unexpected mask values {sorted(unique_vals)} for annotation {annotation_id}"
            )

        out_path = (
            out_dir
            / doctor_id
            / str(patient_id)
            / f"{Path(image_name).stem}__ann{annotation_id}.png"
        )
        out_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_path), mask)

        wound_out_path = (
            wound_out_dir
            / doctor_id
            / str(patient_id)
            / f"{Path(image_name).stem}__ann{annotation_id}.png"
        )
        wound_out_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(wound_out_path), (wound_mask > 0).astype(np.uint8) * 255)

        manifest.append(
            {
                "doctor_id": doctor_id,
                "patient_id": patient_id,
                "task_id": rec.get("task_id"),
                "annotation_id": annotation_id,
                "project_id": rec.get("project_id"),
                "image_name": image_name,
                "image_key": image_key,
                "image_relpath": image_relpath,
                "image_path": image_path,
                "height": h,
                "width": w,
                "wound_polygon_count": len(wound_polygons),
                "wound_mask_path": str(wound_out_path),
                "mask_path": str(out_path),
            }
        )

    if not manifest:
        raise SystemExit("No masks were rasterized.")

    write_json(args.manifest_json, manifest)
    print(f"Rasterized masks: {len(manifest)}")
    print(f"Skipped missing images: {skipped_missing_image}")
    print(f"Filled unlabeled pixels inside wound outline: {total_filled_inside_wound}")
    print(f"Wrote manifest: {args.manifest_json}")


if __name__ == "__main__":
    main()
