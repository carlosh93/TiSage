#!/usr/bin/env python3
"""
Step 4: Select one annotator mask per image using a doctor mapping.

This replaces majority-vote consensus with explicit selection:
- If image_key exists in selection mapping -> use mapped doctor.
- Else if --default-doctor exists for that image -> use that doctor.
- Else fallback to first available annotation (deterministic sort by doctor_id).

Outputs (kept compatible with downstream steps):
- data/LUTS/Masks/**.png
- data/LUTS/Wound_Masks/**.png
- data/LUTS/Annotations/processed/consensus_manifest.json
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from common import VALID_MASK_VALUES, parse_json, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Select one LUTS annotation per image.")
    parser.add_argument(
        "--groups-json",
        default="data/LUTS/Annotations/processed/image_groups.json",
        help="Image groups JSON path (step 3 output).",
    )
    parser.add_argument(
        "--selection-json",
        default="data/LUTS/Annotations/processed/selected_doctor_by_image.json",
        help=(
            "JSON mapping image_key -> doctor_id. Can also be "
            "{'selections': {image_key: doctor_id}}."
        ),
    )
    parser.add_argument(
        "--default-doctor",
        default="user_9",
        help="Fallback doctor if an image_key has no explicit mapping entry.",
    )
    parser.add_argument(
        "--strict-selection",
        action="store_true",
        help="Fail when neither mapped doctor nor default doctor is available for an image.",
    )
    parser.add_argument(
        "--output-dir",
        default="data/LUTS/Masks",
        help="Output directory for selected tissue masks.",
    )
    parser.add_argument(
        "--wound-output-dir",
        default="data/LUTS/Wound_Masks",
        help="Output directory for selected wound masks.",
    )
    parser.add_argument(
        "--manifest-json",
        default="data/LUTS/Annotations/processed/consensus_manifest.json",
        help="Output manifest path (kept compatible with downstream step names).",
    )
    return parser.parse_args()


def _load_selection_map(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    data = parse_json(path)
    if isinstance(data, dict) and "selections" in data and isinstance(data["selections"], dict):
        return {str(k): str(v) for k, v in data["selections"].items()}
    if isinstance(data, dict):
        return {str(k): str(v) for k, v in data.items()}
    raise SystemExit(f"Invalid selection mapping format in {path}")


def _read_mask(path: str) -> np.ndarray:
    arr = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if arr is None:
        raise SystemExit(f"Could not read mask: {path}")
    if arr.ndim != 2:
        raise SystemExit(f"Mask is not single-channel: {path}")
    unique = set(np.unique(arr).tolist())
    if not unique.issubset(VALID_MASK_VALUES):
        raise SystemExit(f"Unexpected values {sorted(unique)} in mask: {path}")
    return arr.astype(np.uint8)


def _read_wound_or_derive(wound_path: str | None, tissue_mask: np.ndarray) -> np.ndarray:
    if wound_path:
        arr = cv2.imread(wound_path, cv2.IMREAD_UNCHANGED)
        if arr is not None and arr.ndim == 2 and np.any(arr > 0):
            return np.where(arr > 0, 255, 0).astype(np.uint8)
    return np.where(tissue_mask > 0, 255, 0).astype(np.uint8)


def main() -> None:
    args = parse_args()

    groups = parse_json(args.groups_json)
    if not isinstance(groups, list) or not groups:
        raise SystemExit("Image groups JSON is empty or invalid.")

    selection_map = _load_selection_map(Path(args.selection_json))

    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    wound_out_dir = Path(args.wound_output_dir).resolve()
    wound_out_dir.mkdir(parents=True, exist_ok=True)

    manifest: list[dict] = []
    n_mapped = 0
    n_default = 0
    n_fallback = 0
    n_missing = 0

    for group in groups:
        image_key = str(group.get("image_key") or "")
        image_name = group.get("image_name")
        patient_id = group.get("patient_id") or "unknown_patient"
        annotations = group.get("annotations", []) or []
        if not image_key or not image_name or not annotations:
            continue

        ann_by_doctor = {}
        for ann in annotations:
            did = str(ann.get("doctor_id") or "")
            if did:
                ann_by_doctor[did] = ann

        selected_doctor = selection_map.get(image_key)
        selected_ann = ann_by_doctor.get(selected_doctor) if selected_doctor else None
        selection_source = "mapping"
        if selected_ann is not None:
            n_mapped += 1
        else:
            if args.default_doctor in ann_by_doctor:
                selected_doctor = args.default_doctor
                selected_ann = ann_by_doctor[selected_doctor]
                selection_source = "default_doctor"
                n_default += 1
            elif args.strict_selection:
                n_missing += 1
                continue
            else:
                # Deterministic fallback for non-golden / missing-doctor images.
                selected_doctor = sorted(ann_by_doctor.keys())[0]
                selected_ann = ann_by_doctor[selected_doctor]
                selection_source = "first_available"
                n_fallback += 1

        mask_path = selected_ann.get("mask_path")
        if not mask_path:
            n_missing += 1
            continue
        tissue = _read_mask(mask_path)
        wound = _read_wound_or_derive(selected_ann.get("wound_mask_path"), tissue)

        patient_dir = out_dir / str(patient_id)
        patient_dir.mkdir(parents=True, exist_ok=True)
        out_path = patient_dir / f"{Path(image_name).stem}.png"
        cv2.imwrite(str(out_path), tissue)

        wound_patient_dir = wound_out_dir / str(patient_id)
        wound_patient_dir.mkdir(parents=True, exist_ok=True)
        wound_out_path = wound_patient_dir / f"{Path(image_name).stem}.png"
        cv2.imwrite(str(wound_out_path), wound)

        manifest.append(
            {
                "image_key": image_key,
                "image_name": image_name,
                "patient_id": patient_id,
                "is_golden_patient": bool(group.get("is_golden_patient")),
                "n_annotators": len(ann_by_doctor),
                "doctor_ids": sorted(ann_by_doctor.keys()),
                "method": "selected_doctor",
                "selection_source": selection_source,
                "selected_doctor_id": selected_doctor,
                "source_image_path": group.get("source_image_path"),
                "mask_path": str(out_path),
                "wound_mask_path": str(wound_out_path),
                "wound_method": "selected_doctor_or_derived",
            }
        )

    if not manifest:
        raise SystemExit("No selected masks were generated.")

    write_json(args.manifest_json, manifest)
    print(f"Selected masks generated: {len(manifest)}")
    print(f"Used explicit mapping: {n_mapped}")
    print(f"Used default doctor ({args.default_doctor}): {n_default}")
    print(f"Used first available fallback: {n_fallback}")
    print(f"Skipped missing/invalid: {n_missing}")
    print(f"Wrote manifest: {args.manifest_json}")


if __name__ == "__main__":
    main()
